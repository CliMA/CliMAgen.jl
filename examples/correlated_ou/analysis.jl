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
    dl, dl_test  = get_data_correlated_ou2d(
        batchsize;
        pairs_per_τ = npairs_per_τ,
        f = 0.1,
        resolution=resolution,
        FT=Float32,
        standard_scaling = standard_scaling,
        read = true,
        save = false,
        preprocess_params_file,
        rng=Random.GLOBAL_RNG
    )
    # compute max/min using map reduce, then generate a single batch = nsamples
    xtest = cat([x for x in dl_test]..., dims=4)
    # To use Images.Gray, we need the input to be between 0 and 1.
    # Obtain max and min here using the whole data set
    maxtest = maximum(xtest, dims=(1, 2, 4))
    mintest = minimum(xtest, dims=(1, 2, 4))
    
    # To compare statistics from samples and testing data,
    # cut testing data to length nsamples.
    nsamples = 100
    xtest = xtest[:, :, :, 1:nsamples]


    # Movie to make sure
    clims= (percentile(xtest[:],0.1), percentile(xtest[:], 99.9))
    anim = convert_to_animation(xtest, 1, clims)
    gif(anim, string("anim_test_data.gif"), fps = 10)
    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    
    # sample from the model
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
        samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c = xtest[:,:,sample_channels+1:end, :] |> device)
    elseif sampler == "pc"
        samples = predictor_corrector_sampler(model, init_x, time_steps, Δt; c = xtest[:,:,sample_channels+1:end, :] |> device)
    end
    samples = cpu(samples)

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtest[:,:,1:sample_channels, :], samples, savedir, "spatial_mean_distribution.png", logger=logger)

    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtest[:,:,1:sample_channels, :], samples, savedir, "qq_plot.png", logger=logger)

    # create plots for comparison of real vs. generated spectra
    spectrum_plot(xtest[:,:,1:sample_channels, :], samples, savedir, "mean_spectra.png", logger=logger)

    # create plots with nimages images of sampled data and testing data
    # Rescale now using mintest and maxtest
    xtest = @. (xtest - mintest) / (maxtest - mintest)
    samples = @. (samples - mintest) / (maxtest - mintest)

    img_plot(samples[:, :, [1], 1:nimages], savedir, "$(sampler)_images_ch1.png")
    img_plot(xtest[:, :, [1], 1:nimages], savedir, "test_images_ch1.png")
    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    
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
