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

function generate_samples(params, μ0, σ0; bias = nothing, k_bias = 0f0, analytic=false,FT=Float32)
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
    samplesdir = joinpath(savedir, "bias_$(analytic)_$(FT(k_bias))")
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


function compute_statistics(tomlfile, xtrain, A, k_bias; analytic=false,FT = Float32)
    # read experiment parameters from file
    params = TOML.parsefile(tomlfile)
    params = CliMAgen.dict2nt(params)

    # unpack params
    savedir = params.experiment.savedir
    samplesdir = joinpath(savedir, "bias_$(analytic)_$(FT(k_bias))")
    samples_file = params.sampling.samples_file
    samples = read_from_hdf5(; hdf5_path = joinpath(samplesdir, samples_file))
    outputdir = samplesdir

    Z = mean(exp.(k_bias .* A(xtrain)))
    lr = Z.*exp.(-k_bias .*A(samples))[:]
    event_probability_plot(A(xtrain)[:], A(samples)[:], lr, samplesdir, "event_probability.png")
    return mean(lr), mean(A(samples)[:]), std(A(samples)[:])
end

function run_analysis_with_k(k, xtrain; experiment_toml="Experiment_gaussian.toml")
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    nogpu = params.experiment.nogpu
    # set up device
    if !nogpu && CUDA.has_cuda()
        dev = Flux.gpu
        @info "Sampling on GPU"
    else
        dev = Flux.cpu
        @info "Sampling on CPU"
    end
    # set up directory for saving checkpoints
    savedir = params.experiment.savedir
    tilesize = size(xtrain)[1]
    inchannels = 1
    # set up bias for space-time mean
    indicator = zeros(FT, tilesize, tilesize, inchannels)
    indicator[8:8, 8:8, :] .= 1
    A(x; indicator = indicator) = sum(indicator .* x, dims=(1, 2, 3)) ./ sum(indicator, dims=(1, 2, 3))
    gpu_indicator = dev(indicator)
    gpu_∂A∂x(x; indicator = gpu_indicator) = indicator ./ sum(indicator, dims=(1, 2, 3))

    #Used in analytic score. All pixels are drawn from the same distribution
    #so use a scalar here.
    σ0 = std(xtrain)
    μ0 = mean(xtrain)
    gpu_bias(x, k = k) = k*gpu_∂A∂x(x)
    generate_samples(params, μ0, σ0; bias =  gpu_bias, k_bias = k, analytic=true,FT=Float32)
    generate_samples(params, μ0, σ0; bias = gpu_bias, k_bias = k, analytic=false,FT=Float32)
    samples_file = params.sampling.samples_file
    analytic_samplesdir = joinpath(savedir, "bias_true_$(FT(k))")
    samplesdir = joinpath(savedir, "bias_false_$(FT(k))")
    samples_analytic = read_from_hdf5(; hdf5_path = joinpath(analytic_samplesdir, samples_file));
    samples = read_from_hdf5(; hdf5_path = joinpath(samplesdir, samples_file));

    Plots.histogram(A(samples_analytic)[:], label = "generated, analytic", title="Event distribution", xlabel = "Pixel value", norm = true)
    Plots.histogram!(A(samples)[:], label = "generated, score", norm = true)
    Plots.histogram!(A(xtrain)[:], label = "training", norm = true, margin = 15Plots.mm)
    Plots.savefig(joinpath(savedir, "event_histogram_$(FT(k)).png"))
    μA = mean(A(xtrain))
    σA = std(A(xtrain))
    expected_mean = μA + σA^2 *k
    expected_sigma = σA
    (na,ma,sa) = compute_statistics(experiment_toml, xtrain, A, k; analytic=true,FT = FT)
    (n,m,s) = compute_statistics(experiment_toml, xtrain, A, k; analytic=false,FT = FT)
    return (expected = (mean = expected_mean, sigma = expected_sigma), analytic = (norm = na, mean = ma, sigma = sa), score = (norm = n, mean = m, sigma = s))
end



function behavior_with_k(experiment_toml="Experiment_gaussian.toml")
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
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
    results = []
    for k in [0f0, 0.4f0, 0.8f0, 1.6f0, 3.2f0]
        stats = run_analysis_with_k(k, xtrain; experiment_toml="Experiment_gaussian.toml")
        push!(results, stats)
    end

    # do something with the stats - error as a function of k
end
