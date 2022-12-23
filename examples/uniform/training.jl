using CUDA
using Dates
using Flux
using Random
using TOML

using CliMAgen
using CliMAgen: dict2nt
using CliMAgen: VarianceExplodingSDE, NoiseConditionalScoreNetworkVariant
using CliMAgen: score_matching_loss_variant
using CliMAgen: WarmupSchedule, ExponentialMovingAverage
using CliMAgen: train!, load_model_and_optimizer

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl")) # for data loading
include(joinpath(package_dir,"examples/utils_wandb.jl")) # for wandb logging, needs correct Python install
include("analysis.jl") # for analysis

function run_training(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    batchsize = params.data.batchsize
    tilesize = params.data.tilesize
    sigma_min::FT = params.model.sigma_min
    sigma_max::FT = params.model.sigma_max
    inchannels = params.model.inchannels
    shift_input = params.model.shift_input
    shift_output = params.model.shift_output
    mean_bypass = params.model.mean_bypass
    scale_mean_bypass = params.model.scale_mean_bypass
    gnorm = params.model.gnorm
    nwarmup = params.optimizer.nwarmup
    gradnorm::FT = params.optimizer.gradnorm
    learning_rate::FT = params.optimizer.learning_rate
    beta_1::FT = params.optimizer.beta_1
    beta_2::FT = params.optimizer.beta_2
    epsilon::FT = params.optimizer.epsilon
    ema_rate::FT = params.optimizer.ema_rate
    nepochs = params.training.nepochs
    freq_chckpt = params.training.freq_chckpt
    std = params.data.std
    ndata = params.data.ndata
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
    dataloaders = get_data_uniform(
        batchsize, std, ndata;
        size = tilesize,
        FT=FT
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
        net = NoiseConditionalScoreNetworkVariant(; 
            inchannels = inchannels,
            shift_input = shift_input,
            shift_output = shift_output,
            mean_bypass = mean_bypass,
            scale_mean_bypass = scale_mean_bypass,
            gnorm = gnorm,
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
    lossfn = x -> score_matching_loss_variant(model, x)

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



function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    # set up directory for saving checkpoints
    !ispath(params.experiment.savedir) && mkpath(params.experiment.savedir)

    # start logging if applicable
    if params.experiment.logging
        logger = Wandb.WandbLogger(
            project=params.experiment.project,
            name="$(params.experiment.name)-$(Dates.now())",
            config=struct2dict(params), # need this otherwise wandb doesn't log the config
        )
    else
        logger = nothing
    end

    run_training(params; FT=FT, logger=logger)

    if :sampling in keys(params)
        run_analysis(params; FT=FT, logger=logger)
    end

    # close the logger after the run to avoid hanging logger
    if params.experiment.logging
        close(logger)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml = ARGS[1])
end
