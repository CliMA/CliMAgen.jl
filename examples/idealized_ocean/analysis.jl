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
using CliMAgen: dict2nt
using CliMAgen: VarianceExplodingSDE, NoiseConditionalScoreNetwork
using CliMAgen: score_matching_loss
using CliMAgen: WarmupSchedule, ExponentialMovingAverage
using CliMAgen: train!, load_model_and_optimizer

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))
using CUDA
using Dates
using Flux
using Random
using TOML
using BSON
using DelimitedFiles

include("ocean_data.jl") # for data loading

function run_analysis(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    train_fraction = params.data.train_fraction
    irange = params.data.i_init:params.data.i_end
    jrange = params.data.j_init:params.data.j_end

    inchannels = params.model.noised_channels
    nsamples = params.sampling.nsamples
    nimages = params.sampling.nimages
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    tilesize_sampling = params.sampling.tilesize

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
    if inchannels != 3
        channels = 1:inchannels
    else
        channels = [1, 2, 4]
    end
    dl = get_data_ocean(batchsize; 
                        channels = channels,
                        irange, 
                        jrange, 
                        train_fraction, 
                        sigma_max_comp = false)
    xtrain = first(dl)

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    
    # sample from the trained model
    time_steps, Δt, init_x = setup_sampler(
        model,
        device,
        tilesize_sampling,
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

    # # create plot showing distribution of spatial mean of generated and real images
    # spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png", logger=logger)

    # # create q-q plot for cumulants of pre-specified scalar statistics
    # qq_plot(xtrain, samples, savedir, "qq_plot.png", logger=logger)

    # # create plots for comparison of real vs. generated spectra
    # spectrum_plot(xtrain, samples, savedir, "mean_spectra.png", logger=logger)

    # create plots with nimages images of sampled data and training data
    for ch in 1:inchannels
        heatmap_grid(samples[:, :, ch:ch, 1:nimages], 1, savedir, "$(sampler)_images_$(ch).png")
        heatmap_grid(xtrain.data.data[:, :, [ch], 1:nimages], 1, savedir, "train_images_$(ch).png")
    end
    
    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    logger = nothing
    run_analysis(params; FT=FT, logger=logger)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
