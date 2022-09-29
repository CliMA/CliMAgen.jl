using CUDA
using Dates
using Flux
using Random

using CliMAgen
using CliMAgen: parse_commandline, dict2nt
using CliMAgen: HyperParameters, VarianceExplodingSDE, DenoisingDiffusionNetwork
using CliMAgen: score_matching_loss
using CliMAgen: WarmupSchedule, ExponentialMovingAverage
using CliMAgen: train!, load_model_and_optimizer

CUDA.allowscalar(false)
include("../utils_data.jl")

function run(args, hparams; FT=Float32, logger=nothing)
    # set up rng
    args.seed > 0 && Random.seed!(args.seed)

    # set up device
    if !args.nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Training on GPU"
    else
        device = Flux.cpu
        @info "Training on CPU"
    end

    # set up dataset
    dataloaders = get_data_cifar10(hparams.data, FT=FT)

    # set up model & optimizer
    if args.restartfile isa Nothing
        #net = NoiseConditionalScoreNetwork(; inchannels = hparams.data.inchannels)
        net = DenoisingDiffusionNetwork(; inchannels = hparams.data.inchannels)
        model = VarianceExplodingSDE(hparams.model; net=net)
        opt = Flux.Optimise.Optimiser(
            WarmupSchedule{FT}(
                hparams.optimizer.nwarmup
            ), 
            Flux.Optimise.ClipNorm(hparams.optimizer.gradclip),
            Flux.Optimise.Adam(
                hparams.optimizer.lr, 
                (hparams.optimizer.β1, hparams.optimizer.β2), 
                hparams.optimizer.ϵ
            )
        )
    else
        @info "Initializing the model, optimizer, and hyperparams from checkpoint."
        @info "Overwriting passed hyperparameters with checkpoint values."
        (; opt, model, hparams) = load_model_and_optimizer(args.restartfile)
    end
    model = device(model)

    # set up moving average for model parameters
    opt_smooth = ExponentialMovingAverage(hparams.optimizer.ema_rate)

    # set up loss function
    lossfn = x->score_matching_loss(model, x)

    # train the model
    train!(
        model, 
        lossfn, 
        dataloaders, 
        opt,
        opt_smooth,
        hparams,
        device,
        args.savedir,
        logger
    )
end

function main(FT=Float32)
    # set arguments for run
    args = parse_commandline() # returns a dictionary which is converted to a NamedTuple
    args = dict2nt(args)

    # hyperparameters
    hparams = HyperParameters(
        data = (;
                nbatch  = 64,
                inchannels = 3,
                ),
        model = (; 
                #  σ_max   = FT(50),
                #  σ_min   = FT(0.01),
                 σ_max   = FT(4.66),
                 σ_min   = FT(0.466),
                 ),
        optimizer = (;     
                     lr      = FT(2e-4),
                     ϵ       = FT(1e-8),
                     β1      = FT(0.9),
                     β2      = FT(0.999),
                     nwarmup = 5000,
                     gradclip = FT(1.0),
                     ema_rate = FT(0.999),
                     ),
        training = (; 
                    nepochs = 4000, 
                    )
    )
    
    run(args, hparams; FT=FT, logger=nothing)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

