using CUDA
using Dates
using Flux
using Random

using CliMAgen
using CliMAgen: parse_commandline, dict2nt
using CliMAgen: HyperParameters, VarianceExplodingSDE, NoiseConditionalScoreNetwork, NoiseConditionalScoreNetworkVariant, DenoisingDiffusionNetwork
using CliMAgen: score_matching_loss, score_matching_loss_variant
using CliMAgen: WarmupSchedule, ExponentialMovingAverage
using CliMAgen: train!, load_model_and_optimizer
    
include("../utils_wandb.jl")
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
    tile_size = hparams.data.size
    dataloaders = get_data_2dturbulence(
        hparams.data;
        width=(tile_size, tile_size),
        stride=(tile_size, tile_size),
        FT=FT
    )

    # set up model & optimizer
    if args.restartfile isa Nothing
        net = NoiseConditionalScoreNetworkVariant(; 
            inchannels=hparams.data.inchannels,
            shift_input=hparams.model.shift_input,
            shift_output=hparams.model.shift_output,
            mean_bypass=hparams.model.mean_bypass,
            scale_mean_bypass=hparams.model.scale_mean_bypass,
        )

        model = VarianceExplodingSDE(hparams.model; net=net)
        opt = Flux.Optimise.Optimiser(
            WarmupSchedule{FT}(
                hparams.optimizer.nwarmup
            ),
            Flux.Optimise.ClipNorm(hparams.optimizer.gradnorm),
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
    lossfn = x -> score_matching_loss(model, x)

    # train the model
    train!(
        model,
        lossfn,
        dataloaders,
        opt,
        opt_smooth,
        hparams,
        device,
        joinpath(args.savedir, "$(Dates.now())"),
        logger,
        hparams.training.freq_chckpt,
    )
end

function main(FT=Float32)
    args = parse_commandline() # returns a dictionary which is converted to a NamedTuple
    args = dict2nt(args)

    # hyperparameters
    hparams = HyperParameters(
        data=(;
            nbatch=16,
            inchannels=2,
            size=256,
        ),
        model=(;
            σ_max=FT(180.0),
            σ_min=FT(0.01),
            mean_bypass=false,
            shift_input=false, 
            shift_output=false,
            scale_mean_bypass=false,
        ),
        optimizer=(;
            lr=FT(0.0002),
            ϵ=FT(1e-8),
            β1=FT(0.9),
            β2=FT(0.999),
            nwarmup=5000,
            gradnorm=FT(1),
            ema_rate=FT(0.999),
        ),
        training=(;
            nepochs=240,
            freq_chckpt=80,
        )
    )

    if args.logging 
        logger = Wandb.WandbLogger(
        project="CliMAgen.jl",
        name="2dturbulence_nx$(hparams.data.size)_nbatch$(hparams.data.nbatch)-all-off-vanilla-loss-$(Dates.now())",
        config=Dict(),
    )
    else
        logger = nothing
    end

    run(args, hparams; FT=FT, logger=logger)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
