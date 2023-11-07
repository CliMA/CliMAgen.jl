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
include(joinpath(package_dir,"examples/utils_etl.jl"))

function run_analysis(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    resolution = params.data.resolution
    wavenumber::FT = params.data.wavenumber
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    inchannels = params.model.inchannels

    make_samples = params.sampling.make_samples
    samples_file = params.sampling.samples_file
    nsamples = params.sampling.nsamples
    nimages = params.sampling.nimages
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler

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
    dl, _ =  get_data_extreme2dturbulence(batchsize;
        resolution = resolution,
        wavenumber = wavenumber,
        fraction = fraction,
        standard_scaling = standard_scaling,
        FT=FT,
        read=true,
        preprocess_params_file=preprocess_params_file
    )
    xtrain = cat([x for x in dl]..., dims=4)

    # To compare statistics from samples and training data,
    # cut training data to length nsamples.
    xtrain = xtrain[:, :, :, 1:nsamples]

    if make_samples
        # set up model
        checkpoint_path = joinpath(savedir, "checkpoint.bson")
        BSON.@load checkpoint_path model model_smooth opt opt_smooth
        model = device(model)
        
        # sample from the trained model
        time_steps, Δt, init_x = setup_sampler(
            model,
            device,
            resolution,
            inchannels;
            num_images=nsamples,
            num_steps=nsteps,
        )
        if sampler == "euler"
            samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
        elseif sampler == "pc"
            samples = predictor_corrector_sampler(model, init_x, time_steps, Δt)
        end
        samples = cpu(samples)
    else
        samples = read_from_hdf5(params, filename=samples_file)
    end


    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png", logger=logger)

    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain, samples, savedir, "qq_plot.png", logger=logger)

    # create plots for comparison of real vs. generated spectra
    spectrum_plot(xtrain, samples, savedir, "mean_spectra.png", logger=logger)

    heatmap_grid(samples[:, :, [1], 1:nimages], 1, savedir, "$(sampler)_images.png")
    heatmap_grid(xtrain[:, :, [1], 1:nimages], 1, savedir, "train_images.png")

    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    run_analysis(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
