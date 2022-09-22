using CUDA
using Dates
using Flux
using Random

using CliMAgen
using CliMAgen: parse_commandline, dict2nt
using CliMAgen: HyperParameters, VarianceExplodingSDE, NoiseConditionalScoreNetwork
using CliMAgen: score_matching_loss
using CliMAgen: WarmupSchedule
using CliMAgen: train!, load_model_and_optimizer

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

    # set up directory structure
    !ispath(args.savedir) && mkpath(args.savedir)

    # set up dataset
    dataloaders = get_data_mnist(hparams.data, FT=FT)

    # set up model & optimizer
    if args.restartfile isa Nothing
        net = NoiseConditionalScoreNetwork()
        model = VarianceExplodingSDE(hparams.model; net=net)
        opt = Flux.Optimise.Optimiser(
            WarmupSchedule{FT}(
                hparams.optimizer.nwarmup
            ),
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

    # set up loss function
    lossfn = x->score_matching_loss(model, x)

    # train the model
    train!(
        model, 
        lossfn, 
        dataloaders, 
        opt, 
        hparams, 
        args, 
        device, 
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
        ),
        model = (; 
            σ_max   = FT(4.66),
            σ_min   = FT(0.466),
        ),
        optimizer = (;     
            lr      = FT(0.0002),
            ϵ       = FT(1e-8),
            β1      = FT(0.9),
            β2      = FT(0.999),
            nwarmup = 1,
        ),
        training = (; 
            nepochs = 30, 
        )
    )

    run(args, hparams; FT=FT, logger=nothing)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

