using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using Revise

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function run_analysis(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    resolution = params.data.resolution
    npairs_per_τ = params.data.npairs_per_tau
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    inchannels = params.model.noised_channels
    nsamples = params.sampling.nsamples
    nimages = params.sampling.nimages
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    tilesize_sampling = params.sampling.tilesize
    sample_channels = params.sampling.sample_channels
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

    # set up dataset
    dl, _  = get_data_correlated_ou2d(
        batchsize;
        pairs_per_τ = npairs_per_τ,
        f = fraction,
        resolution=resolution,
        FT=Float32,
        standard_scaling = standard_scaling,
        read = true,
        save = false,
        preprocess_params_file,
        rng=Random.GLOBAL_RNG
    )
    # compute max/min using map reduce, then generate a single batch = nsamples
    xtrain = cat([x for x in dl]..., dims=4)
    # To use Images.Gray, we need the input to be between 0 and 1.
    # Obtain max and min here using the whole data set
    maxtrain = maximum(xtrain, dims=(1, 2, 4))
    mintrain = minimum(xtrain, dims=(1, 2, 4))
    
    # To compare statistics from samples and training data,
    # cut training data to length nsamples.
    xtrain = xtrain[:, :, :, 1:nsamples]

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    
    # sample from the trained model
    time_steps, Δt, init_x = setup_sampler(
        model,
        device,
        tilesize_sampling,
        sample_channels;
        num_images=nsamples,
        num_steps=nsteps,
    )
    # assert sample channels  = half the total channels?
    if sampler == "euler"
        samples = CliMAgen.Euler_Maruyama_timeseries_sampler(model, init_x, time_steps, Δt; c = xtrain[:,:,sample_channels+1:end, :])
    elseif sampler == "pc"
        samples = predictor_corrector_sampler(model, init_x, time_steps, Δt; c = xtrain[:,:,sample_channels+1:end, :])
    end
    samples = cpu(samples)
    clims= (percentile(samples[:],0.1), percentile(samples[:], 99.9))
    anim = convert_to_animation(samples, 1, clims)
    gif(anim, string("anim.gif"), fps = 10)

   
end

function main(; experiment_toml="Experiment_all_data.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    run_analysis(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
