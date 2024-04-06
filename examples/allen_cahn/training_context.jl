using Flux
using CUDA
using cuDNN
using Dates
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

# run from giorgni2d
include("../utils_data.jl") # for data loading
include("dataloader.jl") # for data loading

function run_training(params, f_path, savedir; FT=Float32)
    # unpack params
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    fraction = params.data.fraction

    sigma_min::FT = params.model.sigma_min
    sigma_max::FT = params.model.sigma_max
    inchannels = params.model.noised_channels
    context_channels = params.model.context_channels
    shift_input = params.model.shift_input
    shift_output = params.model.shift_output
    mean_bypass = params.model.mean_bypass
    scale_mean_bypass = params.model.scale_mean_bypass
    gnorm = params.model.gnorm
    proj_kernelsize = params.model.proj_kernelsize
    outer_kernelsize = params.model.outer_kernelsize
    middle_kernelsize = params.model.middle_kernelsize
    inner_kernelsize = params.model.inner_kernelsize
    dropout_p::FT = params.model.dropout_p
    periodic = params.model.periodic
    
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

    dataloaders = get_data(f_path, "timeseries", batchsize)

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
                                           context = true,
                                           noised_channels = inchannels,
                                           context_channels = context_channels,
                                           shift_input = shift_input,
                                           shift_output = shift_output,
                                           mean_bypass = mean_bypass,
                                           scale_mean_bypass = scale_mean_bypass,
                                           gnorm = gnorm,
                                           dropout_p = dropout_p,
                                           proj_kernelsize = proj_kernelsize,
                                           outer_kernelsize = outer_kernelsize,
                                           middle_kernelsize = middle_kernelsize,
                                           inner_kernelsize = inner_kernelsize,
                                           periodic = periodic,
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
    function lossfn(y; noised_channels = inchannels, context_channels=context_channels)
        x = y[:,:,1:noised_channels,:]
        c = y[:,:,(noised_channels+1):(noised_channels+context_channels),:]
        return score_matching_loss(model, x; c = c)
    end
    

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
        logger = nothing,
        freq_chckpt = freq_chckpt,
    )
end

function main(; model_toml="Model.toml", experiment_toml="Experiment.toml")
    FT = Float32
    toml_dict = TOML.parsefile(model_toml)
    α = FT(toml_dict["param_group"]["alpha"])
    β = FT(toml_dict["param_group"]["beta"])
    γ = FT(toml_dict["param_group"]["gamma"])
    σ = FT(toml_dict["param_group"]["sigma"])
    f_path = "data/data_$(α)_$(β)_$(γ)_$(σ)_context.hdf5"

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    savedir = "$(params.experiment.savedir)_$(α)_$(β)_$(γ)_$(σ)"
    # set up directory for saving checkpoints
    !ispath(savedir) && mkpath(savedir)
    run_training(params, f_path, savedir; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; model_toml = ARGS[1], experiment_toml=ARGS[2])
end