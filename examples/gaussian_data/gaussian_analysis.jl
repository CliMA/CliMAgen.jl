using Flux
using CUDA
using cuDNN
using Dates
using Random
using TOML
using BSON 
using MLUtils
using DataLoaders
using Plots
using ProgressMeter
using Printf
using StatsPlots
using CliMAgen
using CliMAgen: dict2nt,load_model_and_optimizer

package_dir = pkgdir(CliMAgen)
include("../utils_etl.jl")
include("gaussian_data.jl")

function Euler_Maruyama_ld_sampler_analytic_score(model::CliMAgen.AbstractDiffusionModel,
                                                    init_x::A,
                                                    time_steps,
                                                    Δt, μ0, σ0;
                                                    bias=nothing,
                                                    use_shift=false,
                                                    c=nothing,
                                                    forward = false,
                                                    )::A where {A}
    x = mean_x = init_x
    # Preallocate
    score = similar(x)
    z = similar(x)
    if ~(bias isa Nothing)
        bias_drift = similar(x)
        shift = similar(x)
    end

    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)
        Dt = model.σ_min * (model.σ_max/model.σ_min)^time_step
        if forward
            x .= x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(z)
        else
            if bias isa Nothing 
                @. score = -(x-μ0)/(σ0^2 + Dt^2)
            else
                bias_drift .= bias(x)
                if use_shift
                    @. shift = Dt^2 * bias_drift
                else
                    shift .= eltype(x)(0)
                end
                @. score = -(x+shift-μ0)/(σ0^2 + Dt^2)+ bias_drift
            end
            mean_x .= x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt
            x .= mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(z)
        end
    end
    return x
end

function generate_samples(params, μ0, σ0, filename; bias = nothing, k_bias = 0f0, analytic=false,FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    batchsize = params.data.batchsize
    inchannels = params.model.inchannels
    nsteps = params.sampling.nsteps
    nsamples = params.sampling.nsamples
    samples_file = params.sampling.samples_file
    tilesize = 16


    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        @info "Sampling on CPU"
    end

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)

    # sample from the trained model
    samples_per_batch = batchsize
    nbatch = div(nsamples, samples_per_batch)
    all_samples = zeros(FT, (tilesize, tilesize, inchannels,nbatch*samples_per_batch))
    samples = zeros(FT, (tilesize, tilesize, inchannels,samples_per_batch)) |> device
    for b in 1:nbatch
        time_steps, Δt, init_x = setup_sampler(
            model,
            device,
            tilesize,
            inchannels;
            num_images=samples_per_batch,
            num_steps=nsteps,
        )
        if analytic
            samples .= Euler_Maruyama_ld_sampler_analytic_score(model, init_x, time_steps, Δt, μ0, σ0, bias=bias, use_shift = true)
        else
            samples .= Euler_Maruyama_ld_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(b), bias=bias, use_shift = true)
        end
        all_samples[:,:,:,(b-1)*samples_per_batch+1:b*samples_per_batch] .= cpu(samples)
    end
    samplesdir = joinpath(savedir, filename)
    !ispath(samplesdir) && mkpath(samplesdir)
    drop_to_hdf5(all_samples; hdf5_path=joinpath(samplesdir, samples_file), key = "generated_samples")
end

function event_probability(a_m::Vector{FT},
    lr::Vector{FT}
    ) where {FT<:AbstractFloat}
    sort_indices = reverse(sortperm(a_m))
    a_sorted = a_m[sort_indices]
    lr_sorted = lr[sort_indices] 
    M = length(a_m)
    # γa = P(X > a)
    γ = cumsum(lr_sorted)./M
    # Compute uncertainty 
    γ² = cumsum(lr_sorted.^2.0)./M
    σ_γ = sqrt.(γ² .-  γ.^2.0)/sqrt(M)
    return a_sorted, γ, σ_γ
end
function event_probability_plot(train::Vector{FT}, gen::Vector{FT}, lr_gen, savepath, plotname; logger=nothing) where {FT}
    plts = []
    logscales = [true, false]
    for logscale in logscales
        em, γ, σ_γ = event_probability(gen, lr_gen)
        plt1 = Plots.plot()
        Plots.plot!(plt1, em, γ,  ribbon = (σ_γ, σ_γ), label = "Generated/IS", legend = :bottomleft)
        if logscale
         Plots.plot!(plt1, yaxis = :log10, ylim = [1e-6, 1])
        end

        N_gen = length(gen)
        lr_train = ones(FT, length(train))

        em, γ, σ_γ = event_probability(train[1:N_gen], lr_train[1:N_gen])
        Plots.plot!(plt1, em, γ,  ribbon = (σ_γ, σ_γ), label = "DS")

        em, γ, σ_γ = event_probability(train, lr_train)
        Plots.plot!(plt1, em, γ,  ribbon = (σ_γ, σ_γ), label = "Truth", ylabel = "Probability", xlabel = "Event magnitude", margin = 10Plots.mm)
        push!(plts, plt1)
    end
    plt = Plots.plot(plts..., layout = (1,2))
    Plots.savefig(plt, joinpath(savepath, plotname))
end


function compute_statistics(samples, xtrain, A, k_bias)
    Z = mean(exp.(k_bias .* A(xtrain)))
    lr = Z.*exp.(-k_bias .*A(samples))[:]
    return mean(lr), mean(A(samples)[:]), std(A(samples)[:])
end





function main(experiment_toml; analytic=false, sample=true)
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    nogpu = params.experiment.nogpu
    # set up device
    if !nogpu && CUDA.has_cuda()
        dev = Flux.gpu
    else
        dev = Flux.cpu
    end
    # obtain training data
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    preprocess_params_file = joinpath(savedir, "preprocessing.jld2")
    dataloaders = get_data_gaussian(batchsize,preprocess_params_file;
                                    tilesize = 16,
                                    read =true,
                                    save=false,
                                    FT=FT
                                    )
    xtrain = cat([x for x in dataloaders[1]]..., dims=4);
    σ0 = std(xtrain)
    μ0 = mean(xtrain)
    
    # Set up bias function
    tilesize = size(xtrain)[1]
    inchannels = 1
    indicator = zeros(FT, tilesize, tilesize, inchannels)
    indicator[8:8, 8:8, :] .= 1
    A(x; indicator = indicator) = sum(indicator .* x, dims=(1, 2, 3)) ./ sum(indicator, dims=(1, 2, 3))
    n_pixels = Int(sum(indicator, dims=(1, 2, 3))[1])
    # generate samples
    μA = mean(A(xtrain)[:])
    σA = std(A(xtrain)[:])
    sigma_values = [0f0, 0.5f0, 1.0f0, 1.5f0]
    kvalues = sigma_values ./ σA
    if sample
        gpu_indicator = dev(indicator)
        gpu_∂A∂x(x; indicator = gpu_indicator) = indicator ./ sum(indicator, dims=(1, 2, 3))
        for i in 1:length(kvalues)
            k = kvalues[i]
            filename = "bias_$(analytic)_$(sigma_values[i])_$(n_pixels)"
            gpu_bias(x, k = k) = k*gpu_∂A∂x(x)
            generate_samples(params, μ0, σ0, filename; bias =  gpu_bias, k_bias = k, analytic=analytic,FT=FT)
        end
    end
    # Now, read in samples and compute A on them
    expected_mean = @. μA + σA^2 *kvalues
    expected_sigma = σA
    nsamples = 1280
    observables = zeros(nsamples, length(kvalues))
    for i in 1:length(kvalues)
        k = kvalues[i]
        filename = "bias_$(analytic)_$(sigma_values[i])_$(n_pixels)"
        samples_file = params.sampling.samples_file
        samplesdir = joinpath(savedir, filename)
        samples = read_from_hdf5(; hdf5_path = joinpath(samplesdir, samples_file));
        observables[:, i] .= A(samples)[:]
    end
    violin(sigma_values[1] .+ zeros(nsamples), observables[:,1], linewidth=0,side=:right, label="Gen", color = "red")
    violin!(sigma_values[1] .+ zeros(nsamples), randn(nsamples)*expected_sigma .+ expected_mean[1], linewidth=0,side=:left, label="Expected", color = "blue")
    for i in 2:length(kvalues)
        violin!(sigma_values[i].+ zeros(nsamples), observables[:,i], linewidth=0,side=:right, label = "", color = "red")
        violin!(sigma_values[i] .+ zeros(nsamples), randn(nsamples)*expected_sigma .+ expected_mean[i], linewidth=0,side=:left, label = "", color = "blue")
    end
    plot!(xlabel = "Shift [units of σA]", ylabel = "Value of A(x)", margins = 10Plots.mm)
    savefig(joinpath(savedir, "violin_$(n_pixels).png"))
end
