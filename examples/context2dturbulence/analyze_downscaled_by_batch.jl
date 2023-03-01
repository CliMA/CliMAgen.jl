using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function unpack_experiment(experiment_toml, wavenumber; device = Flux.gpu, FT=Float32)
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    # unpack params
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    scaling = JLD2.load_object(preprocess_params_file)
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels

    # set up dataset - we need this in order to get the context
    dl, _ =  get_data_context2dturbulence(
        batchsize;
        resolution = resolution,
        wavenumber = wavenumber,
        fraction = fraction,
        standard_scaling = standard_scaling,
        FT=FT,
        read=true,
        preprocess_params_file=preprocess_params_file
    )
    train = cat([x for x in dl]..., dims=4)

    xtrain = train[:,:,1:noised_channels,:] |> device
    ctrain = train[:,:,(noised_channels+1):(noised_channels+context_channels),:] |> device

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    return model, xtrain, ctrain, scaling
end

function generate_samples!(samples, init_x, model, context, σ_T, time_steps, Δt, sampler)
    FT = eltype(samples)
    init_x .= randn!(init_x) .* σ_T
    random_selection = StatsBase.sample(1:1:size(context)[end], size(samples)[end])
    # sample from the trained model
    if sampler == "euler"
        samples .= Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=context[:,:,:,random_selection])
    elseif sampler == "pc"
        samples .= predictor_corrector_sampler(model, init_x, time_steps, Δt; c=context[:,:,:,random_selection])
    end
    return samples
end

function main(nbatches, wavenumber; source_toml="Experiment.toml", target_toml="Experiment.toml")
    FT = Float32
    device = Flux.gpu
    nsteps = 125
    savepath = "./"
    nsamples = 10
    channel = 1
    tilesize = 512
    context_channels = 1
    noised_channels = 2

    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, wavenumber; device = device, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, wavenumber; device = device,FT=FT)

    # Determine which `k` the two sets of images begin to differ from each other
    source_spectra, k = batch_spectra(xsource |> cpu, size(xsource)[1])
    target_spectra, k = batch_spectra(xtarget |> cpu, size(xtarget)[1])

    # this is manual right now. just eyeball it.
    cutoff_idx = 3
    k_cutoff = FT(k[cutoff_idx])

    source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))
    forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, forward_model.σ_max, forward_model.σ_min))
    target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, reverse_model.σ_max, reverse_model.σ_min))

    samples = zeros(FT, (tilesize, tilesize, context_channels+noised_channels, nsamples)) |> device

    init_x_forward =  zeros(FT, (tilesize, tilesize, noised_channels, nsamples)) |> device
    t_forward = zeros(FT, nsamples) .+ forward_t_end |> device
    _, σ_forward = CliMAgen.marginal_prob(forward_model, init_x_forward, t_forward)
    time_steps_forward = LinRange(FT(1.0f-5),FT(forward_t_end), nsteps)
    Δt_forward = time_steps_forward[2] - time_steps_forward[1]

    init_x_reverse =  zeros(FT, (tilesize, tilesize, noised_channels, nsamples)) |> device
    t_reverse = zeros(FT, nsamples) .+ reverse_t_end |> device
    _, σ_reverse = CliMAgen.marginal_prob(reverse_model, init_x_reverse, t_reverse)
    time_steps_reverse = LinRange(FT(reverse_t_end), FT(1.0f-5), nsteps)
    Δt_reverse = time_steps_reverse[1] - time_steps_reverse[2]

    filenames = [joinpath(savedir, "downscale_gen_statistics_ch1_$wavenumber.csv"),
                 joinpath(savedir, "downscale_gen_statistics_ch2_$wavenumber.csv")]
    indices = 1:1:size(csource)[end]
    for batch in 1:nbatches
        selection = StatsBase.sample(indices, nsamples)
        init_x_forward .= randn!(init_x_forward) .* σ_forward
        forward_sol = diffusion_simulation(forward_model, init_x_forward, nsteps;
                                            c=csource[:,:,:,selection],
                                            reverse=false,
                                            FT=FT,
                                            ϵ=1.0f-5,
                                            sde=false,
                                            solver=DifferentialEquations.RK4(),
                                            t_end=forward_t_end,
                                            nsave=2)
        init_x_reverse .= forward_sol.u[end]
       # init_x_reverse .= generate_samples!(init_x_reverse,
       #                                     init_x_forward,
       #                                     forward_model,
       #                                     csource[:,:,:,selection],
       #                                     σ_forward,
       #                                     time_steps_forward,
       #                                     Δt_forward,
       #                                     sampler)
        init_x_reverse .= adapt_x!(init_x_reverse, forward_model, reverse_model, forward_t_end, reverse_t_end)
                                    
        samples[:,:,1:noised_channels,:] .= generate_samples!(samples[:,:,1:noised_channels,:],
                                                              init_x_reverse,
                                                              reverse_model,
                                                              ctarget[:,:,:,selection],
                                                              σ_reverse,
                                                              time_steps_reverse,
                                                              Δt_reverse,
                                                              sampler)
        

        # Carry out the inverse preprocessing transform to go back to real space
        samples = invert_preprocessing(cpu(cat(reverse_solution.u[end], ctarget, dims = 3)), scaling_target) |> device

        # compute metrics of interest
        sample_means =  mapslices(Statistics.mean, cpu(samples), dims=[1, 2])
        sample_κ2 = Statistics.var(cpu(samples), dims = (1,2))
        sample_κ3 = mapslices(x -> StatsBase.cumulant(x[:],3), cpu(samples), dims=[1, 2])
        sample_κ4 = mapslices(x -> StatsBase.cumulant(x[:],4), cpu(samples), dims=[1, 2])
        sample_spectra = mapslices(x -> hcat(power_spectrum2d(x, 512)[1]), cpu(samples), dims =[1,2])

        # average instant
        sample_icr = make_icr(cpu(samples))

        #save the metrics
        for ch in 1:noised_channels
            if ch == 1
                output = hcat(sample_means[1,1,ch,:],sample_κ2[1,1,ch,:], sample_κ3[1,1,ch,:],sample_κ4[1,1,ch,:], transpose(sample_spectra[:,1,ch,:]), sample_icr[1,1,ch,:])
            else
                output = hcat(sample_means[1,1,ch,:],sample_κ2[1,1,ch,:], sample_κ3[1,1,ch,:],sample_κ4[1,1,ch,:], transpose(sample_spectra[:,1,ch,:]))
            end
            open(filenames[ch], "a") do io
                writedlm(io, output, ',')
            end
        end
    end
    
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse(Int64, ARGS[1]), parse(Float32, ARGS[2]); experiment_toml=ARGS[3])
end