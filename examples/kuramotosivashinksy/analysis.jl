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

function run_analysis(params; FT=Float32)

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
    xtrain = cat([x for x in dl]..., dims=4);
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
        outputdir = savedir
        lr = ones(FT, nsamples)
    else
        k_bias::FT = params.sampling.k_bias
        shift = params.sampling.shift
        samplesdir = joinpath(savedir, "bias_$(k_bias)_shift_$shift")
        samples_file = params.sampling.samples_file
        samples = read_from_hdf5(; hdf5_path = joinpath(samplesdir, samples_file))
        outputdir = samplesdir


    
        # set up bias for space-time mean
        indicator = zeros(FT, n_pixels, n_time, inchannels)
        indicator[div(n_pixels, 4):3*div(n_pixels, 4), div(n_time, 4):3*div(n_time, 4), :] .= 1
        indicator = device(indicator)

        A(x; indicator = indicator) = sum(indicator .* x, dims=(1, 2, 3)) ./ sum(indicator, dims=(1, 2, 3))
        # Compute normalization using all of the data
        Z = mean(exp.(k_bias .* A(device(xtrain))))
        lr = Z.*exp.(-k_bias .*A(samples; indicator = cpu(indicator)))[:]
    end

    # Autocorrelation code 
    # Expects a timeseries of of a scalar: of size nsteps x nbatch
    # Restrict to the first nsamples so that the uncertainties are comparable
    autocorrelation_plot(xtrain[64,:,1,1:nsamples], samples[64,:,1,:], outputdir, "autocorr.png")
    duration = 16 # autocorrelation time is 10
    observable(x) = mean(x[64,64-div(duration,2):64+div(duration,2)-1,1,:], dims = 1)[:]
    event_probability_plot(observable(xtrain), observable(samples), lr, outputdir, "event_probability_$(duration).png")

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples, outputdir, "spatial_mean_distribution.png")
    
    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain[:,:,:, 1:size(samples)[end]], samples, outputdir, "qq_plot.png")

    # create plots with nimages images of sampled data and training data
    # Rescale now using mintrain and maxtrain
    xtrain = @. (xtrain - mintrain) / (maxtrain - mintrain)
    samples = @. (samples - mintrain) / (maxtrain - mintrain)

    img_plot(samples[:, :, [1], 1:nimages], outputdir, "$(sampler)_images_ch1.png")
    img_plot(xtrain[:, :, [1], 1:nimages], outputdir, "train_images_ch1.png")
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
