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
include(joinpath(package_dir,"examples/correlated_ou_1d/store_load_samples.jl"))

function run_analysis(params; FT=Float32, logger=nothing)

    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
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
    dl,_ = get_data_correlated_ou1d(batchsize,preprocess_params_file;
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
    autocorrelation_plot(xtrain[32,:,1,1:nsamples], samples[32,:,1,:], savedir, "autocorr.png";logger=logger)

    # Return curve for the following metric: the mean of the middle pixel,
    # taken over a block of time length 8
    m = 8
    metric(x) = mean(x[32,32-div(m,2):32+div(m,2)-1,1,:], dims = 1)[:]
    event_probability_plot(metric(xtrain), metric(samples), ones(FT, nsamples), savedir, "event_probability_$m.png"; logger=logger)

    # Im not sure about the following: 
    # To compute the return time, we need more care. We need a time interval associated with this event in 
    # order to turn a probability into a return time. If the block length is longer than the autocorrelation
    # time, I think we can use the block length directly: within each sample of length n_time > m, we get one
    # independent sample of length m. If instead we had carried out a direct numerical simulation
    # and split it into blocks of length m, we would get the same result because the blocks would be independent.

    # The issue arises if the block is shorter than the autocorrelation time. In this case, if we 
    # had carried out a direct numerical simulation and split it into blocks of length m,
    # the blocks would no longer be independent. Since these do not agree, I dont think it's the
    # right thing to do.

    # We could try: min(autocorrelation time, block length), or split each sample into many blocks
    # of length m and take the maximum. Then even if they are correlated, we return an independent 
    # sample per n_time.
    metric_return_time(y) = maximum(mapslices(x -> block_applied_func(x, mean, m), y, dims = 1), dims = 1)[:]
    return_curve_plot(metric_return_time(xtrain[:,:,:,1:nsamples]), metric_return_time(samples), FT(n_time), savedir, "return_curve_$m.png"; logger=logger)

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
