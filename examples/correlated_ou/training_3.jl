using HDF5

function get_data_correlated_ou2d_2(batchsize;
    f = 1.0,
    resolution=32,
    FT=Float32,
    standard_scaling = false,
    read = false,
    save = false,
    preprocess_params_file,
    rng=Random.GLOBAL_RNG,
    shuffle = true)
    
    # @assert xor(read, save)

    # hfile = h5open("/home/sandre/Repositories/CliMAgen.jl/examples/correlated_ou/xdata.hdf5")
    hfile = h5open("/home/sandre/Repositories/CliMAgen.jl/allen_cahn.hdf5")
    x = HDF5.read(hfile, "data")
    x = reshape(x, (32, 32, 1, size(x)[end]))
    N = size(x)[end]
    close(hfile)

    xtrain = FT.(x[:,: , :, 1:round(Int, 0.5*N)])
    xtest = FT.(x[:,: , :, round(Int, 0.5*N)+1:end]) 

    if shuffle
        xtrain = MLUtils.shuffleobs(rng, xtrain)
    end
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end


using CUDA
using Dates
using Flux
using Random
using TOML
using BSON
using DelimitedFiles

using CliMAgen
using CliMAgen: dict2nt
using CliMAgen: VarianceExplodingSDE, NoiseConditionalScoreNetwork
using CliMAgen: score_matching_loss
using CliMAgen: WarmupSchedule, ExponentialMovingAverage
using CliMAgen: train!, load_model_and_optimizer

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl")) # for data loading
include("analysis.jl") # for analysis

function run_training(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    sigma_min::FT = params.model.sigma_min
    sigma_max::FT = params.model.sigma_max
    inchannels = params.model.noised_channels
    shift_input = params.model.shift_input
    shift_output = params.model.shift_output
    mean_bypass = params.model.mean_bypass
    scale_mean_bypass = params.model.scale_mean_bypass
    gnorm = params.model.gnorm
    proj_kernelsize = params.model.proj_kernelsize
    outer_kernelsize = params.model.outer_kernelsize
    middle_kernelsize = params.model.middle_kernelsize
    inner_kernelsize = params.model.inner_kernelsize
    dropout_p = params.model.dropout_p

    nwarmup = params.optimizer.nwarmup
    gradnorm::FT = params.optimizer.gradnorm
    learning_rate::FT = params.optimizer.learning_rate
    beta_1::FT = params.optimizer.beta_1
    beta_2::FT = params.optimizer.beta_2
    epsilon::FT = params.optimizer.epsilon
    ema_rate::FT = params.optimizer.ema_rate

    nepochs = params.training.nepochs
    freq_chckpt = params.training.freq_chckpt

    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Training on GPU"
    else
        device = Flux.cpu
        @info "Training on CPU"
    end

    # set up dataset
    dataloaders = get_data_correlated_ou2d_2(
        batchsize;
        f = fraction,
        resolution=resolution,
        FT=Float32,
        standard_scaling = standard_scaling,
        read = false,
        save = true,
        preprocess_params_file,
        rng=Random.GLOBAL_RNG
    )

    # set up model and optimizers
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    loss_file = joinpath(savedir, "losses.txt")

    if isfile(checkpoint_path) && isfile(loss_file)
        BSON.@load checkpoint_path model model_smooth opt opt_smooth
        model = device(model)
        model_smooth = device(model_smooth)
        loss_data = DelimitedFiles.readdlm(loss_file, ',', skipstart = 1)
        start_epoch = loss_data[end,1]+1
    else
        net = NoiseConditionalScoreNetwork(;
                                           noised_channels = inchannels,
                                           shift_input = shift_input,
                                           shift_output = shift_output,
                                           mean_bypass = mean_bypass,
                                           scale_mean_bypass = scale_mean_bypass,
                                           gnorm = gnorm,
                                           proj_kernelsize = proj_kernelsize,
                                           outer_kernelsize = outer_kernelsize,
                                           middle_kernelsize = middle_kernelsize,
                                           inner_kernelsize = inner_kernelsize,
                                           dropout_p = dropout_p
                                           )
        model = VarianceExplodingSDE(sigma_max, sigma_min, net)
        model = device(model)
        model_smooth = deepcopy(model)

        opt = Flux.Optimise.Optimiser(
            WarmupSchedule{FT}(
                nwarmup 
            ),
            Flux.Optimise.ClipNorm(gradnorm),
            Flux.Optimise.Adam(
                learning_rate,
                (beta_1, beta_2),
                epsilon
            )
        )
        opt_smooth = ExponentialMovingAverage(ema_rate)

        # set up loss file
        loss_names = reshape(["#Epoch", "Mean Train", "Spatial Train","Mean Test","Spatial Test"], (1,5))
        open(loss_file,"w") do io
             DelimitedFiles.writedlm(io, loss_names,',')
        end

        start_epoch=1
    end

    # set up loss function
    lossfn = x -> score_matching_loss(model, x)

    # train the model
    train!(
        model,
        model_smooth,
        lossfn,
        dataloaders,
        opt,
        opt_smooth,
        nepochs,
        device;
        start_epoch = start_epoch,
        savedir = savedir,
        logger = logger,
        freq_chckpt = freq_chckpt,
    )
end

function main(; experiment_toml="Experiment_dropout.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    # set up directory for saving checkpoints
    !ispath(params.experiment.savedir) && mkpath(params.experiment.savedir)

    # start logging if applicable
    logger = nothing

    run_training(params; FT=FT, logger=logger)

    if :sampling in keys(params)
        run_analysis_2(params; FT=FT, logger=logger)
    end

    # close the logger after the run to avoid hanging logger
    if params.experiment.logging
        close(logger)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml = ARGS[1])
end

function run_analysis_2(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    
    resolution = params.data.resolution
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    # preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    preprocess_params_file = false
    
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
    dl, _ = get_data_correlated_ou2d_2(
        nsamples;
        f = fraction,
        resolution=resolution,
        FT=Float32,
        standard_scaling = standard_scaling,
        read = true,
        save = false,
        preprocess_params_file,
        rng=Random.GLOBAL_RNG
    )
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
