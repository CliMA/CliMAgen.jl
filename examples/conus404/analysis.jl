using BSON
using Flux
using CUDA
using cuDNN
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/conus404/preprocessing_utils.jl"))
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function run_analysis(params; FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    batchsize = params.data.batchsize
    standard_scaling = params.data.standard_scaling
    # always use the preprocessing parameters derived 
    # from the training data for this step
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_train.jld2")
    inchannels = params.model.inchannels
    nsamples = params.sampling.nsamples_analysis
    nimages = params.sampling.nimages
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    n_pixels = params.data.n_pixels
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
    dl, _ = get_data_conus404(
        batchsize;
        FT=FT,
        preprocess_params_file=preprocess_params_file
    )
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
        n_pixels,
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

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png")

    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain, samples, savedir, "qq_plot.png")

    # create plots for comparison of real vs. generated spectra
    spectrum_plot(xtrain, samples, savedir, "mean_spectra.png")

    # create plots with nimages images of sampled data and training data
    # Rescale now using mintrain and maxtrain
    xtrain = @. (xtrain - mintrain) / (maxtrain - mintrain)
    samples = @. (samples - mintrain) / (maxtrain - mintrain)

    heatmap_grid(samples[:, :, [1], 1:nimages], 1, savedir, "$(sampler)_images_ch1.png")
    heatmap_grid(xtrain[:, :, [1], 1:nimages], 1, savedir, "train_images_ch1.png")
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
