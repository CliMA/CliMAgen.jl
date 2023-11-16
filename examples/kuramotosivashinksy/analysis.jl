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
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))
include(joinpath(package_dir,"examples/utils_etl.jl"))

function run_analysis(params; FT=Float32, logger=nothing)

    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.sampling.nsamples
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    n_pixels = params.data.n_pixels
    n_time = params.data.n_time
    @assert n_pixels == n_time
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
    dl,_ = get_data_ks(batchsize,preprocess_params_file;
                       f=fraction,
                       n_pixels=n_pixels,
                       n_time=n_time,
                       standard_scaling=standard_scaling,
                       read =true,
                       save=false,
                       FT=FT
                       )
    xtrain = cat([x for x in dl]..., dims=4)
    # To use Images.Gray, we need the input to be between 0 and 1.
    # Obtain max and min here using the whole data set
    maxtrain = maximum(xtrain, dims=(1, 2, 4))
    mintrain = minimum(xtrain, dims=(1, 2, 4))


    if make_samples
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
    else
        samples = read_from_hdf5(samples_file)
    end

    # Autocorrelation code 
    # Expects a timeseries of of a scalar: of size nsteps x nbatch
    # Restrict to the first nsamples so that the uncertainties are comparable
    autocorrelation_plot(xtrain[64,:,1,1:nsamples], samples[64,:,1,:], savedir, "autocorr.png";logger=logger)
    duration = 16 # autocorrelation time is 10
    observable(x) = mean(x[64,64-div(duration,2):64+div(duration,2)-1,1,:], dims = 1)[:]
    lr = ones(FT, nsamples) # no biasing, likelihood ratio is 1.
    event_probability_plot(observable(xtrain), observable(samples), lr, savedir, "event_probability_$duration.png"; logger=logger)

    # To compute the return time, we need more care. We need a time interval associated with this event in 
    # order to turn a probability into a return time. 
    
    # It would be wrong to use the length of the timeseries per sample, because our metric only
    # extracted one event from each sample. The event could happen more than once per sample.

    # If the event duration is much longer than the autocorrelation time, I think we could use the event duration
    # directly: within each sample of length n_time > duration >> autocorrelation time, we could extract one
    # independent sample of length duration, as we do above.
    # If instead we had carried out a direct numerical simulation
    # and split it into blocks of length duration, we would get the same result because the blocks would be ~independent.

    # The issue arises if the event duration is comparable to or shorter than the autocorrelation time. 
    # In this case, if we 
    # had carried out a direct numerical simulation and split it into blocks of length duration,
    # the blocks would no longer be fully independent. Since these do not agree, I dont think it's the
    # right thing to do.

    # Instead, follow Ragone: split each sample into many blocks
    # of length duration and take the maximum. Then even if they are correlated, we return an independent 
    # sample per n_time.
    metric_return_time(y) = maximum(mapslices(x -> block_applied_func(x, mean, duration), y[64, :, 1, :], dims = 1), dims = 1)[:]
    return_curve_plot(metric_return_time(xtrain), metric_return_time(samples), FT(n_time), savedir, "return_curve_$duration.png"; logger=logger)

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png", logger=logger)
    
    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain[:,:,:, 1:nsamples], samples, savedir, "qq_plot.png", logger=logger)

    # create plots with nimages images of sampled data and training data
    # Rescale now using mintrain and maxtrain
    xtrain = @. (xtrain - mintrain) / (maxtrain - mintrain)
    samples = @. (samples - mintrain) / (maxtrain - mintrain)

    img_plot(samples[:, :, [1], 1:nimages], savedir, "$(sampler)_images_ch1.png")
    img_plot(xtrain[:, :, [1], 1:nimages], savedir, "train_images_ch1.png")
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
