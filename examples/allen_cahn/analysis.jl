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

function run_analysis(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    
    split = params.data.split
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    
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
    dl, _ = get_data_allen_cahn(
        nsamples;
        f = fraction,
        split=split,
        FT=Float32,
        standard_scaling = standard_scaling,
        read = true,
        save = false,
        preprocess_params_file,
        rng=Random.GLOBAL_RNG
    )
    #xtrain = first(dl)

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    
    function dyadic!(product, x, s)
        npixels = prod(size(x)[1:3])
        nbatch = size(x)[end]
        s_flat = reshape(s, (npixels, nbatch))
        x_flat = reshape(x, (npixels, nbatch))
        for idx in 1:nbatch
            product .+= x_flat[:,idx] * transpose(s_flat[:,idx])
        end
    end
    # compute expectation of <x s^T>
    npixels = prod(size(x)[1:3])
    product = zeros(FT, (npixels, npixels))
    ndata = 0
    @showprogress "Loop over batches" for x in dl
        s = CliMAgen.score(model, x, eps(FT))
        dyadic!(product, x, s)
        ndata +=1
    end
    Plots.heatmap(product ./ ndata)

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

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png", logger=logger)

    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain, samples, savedir, "qq_plot.png", logger=logger)

    # create plots for comparison of real vs. generated spectra
    spectrum_plot(xtrain, samples, savedir, "mean_spectra.png", logger=logger)

    # create plots with nimages images of sampled data and training data
    for ch in 1:inchannels
        heatmap_grid(samples[:, :, [ch], 1:nimages], ch, savedir, "$(sampler)_images_$(ch).png")
        heatmap_grid(xtrain[:, :, [ch], 1:nimages], ch, savedir, "train_images_$(ch).png")
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
