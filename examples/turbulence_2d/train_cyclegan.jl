using ArgParse
using BSON: @save, @load
using CUDA
using Dates
using Flux
using Flux: params, update!, loadmodel!
using FluxTraining
using HDF5
using Logging
using MLUtils
using ProgressBars
using Statistics: mean
using Wandb

using Downscaling
using Downscaling: PatchDiscriminator2D, UNetGenerator2D, OptimizerHyperParams, create_optimizer

# command line utilities
include(joinpath(pkgdir(Downscaling), "examples", "utils_argparse.jl"))
PARGS = parse_commandline()

# get commandline arguments
DATADIR = PARGS["datadir"]
OUTDIR = PARGS["outdir"]
RESTARTFILE = PARGS["restartfile"]

# specify experiment metadata
EXAMPLE_NAME = "turbulence_2d/cyclegan"
TIME = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")

# set up storage directory for experiment
OUTPUT_DIR = joinpath(OUTDIR, EXAMPLE_NAME, TIME)
mkpath(OUTPUT_DIR)

# get access to the datasets and dataloaders
include(joinpath(DATADIR, "utils_data.jl"))

# set up logger
if PARGS["logging"]
    lg = WandbLogger(project="Superresolution",
        name=joinpath(EXAMPLE_NAME, TIME),
        config=Dict())
    global_logger(lg)
end

# fix float type
const FT = Float32

Base.@kwdef struct LossHyperParams{FT}
    λ = FT(10.0)
    λid = FT(5.0)
end

Base.@kwdef struct TrainingParams
    nepochs::Int = 2
    batch_size::Int = 1
end

function generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, lossparams)
    a_fake = generator_B(b, noise) # Fake image generated in domain A
    b_fake = generator_A(a, noise) # Fake image generated in domain B

    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    a_fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real

    gen_A_loss = mean((a_fake_prob .- 1) .^ 2)
    rec_A_loss = mean(abs.(b - generator_A(a_fake, noise))) # Cycle-consistency loss for domain B
    idt_A_loss = mean(abs.(generator_A(b, noise) .- b)) # Identity loss for domain B
    gen_B_loss = mean((b_fake_prob .- 1) .^ 2)
    rec_B_loss = mean(abs.(a - generator_B(b_fake, noise))) # Cycle-consistency loss for domain A
    idt_B_loss = mean(abs.(generator_B(a, noise) .- a)) # Identity loss for domain A

    return gen_A_loss + gen_B_loss + lossparams.λ * (rec_A_loss + rec_B_loss) + lossparams.λid * (idt_A_loss + idt_B_loss)
end

function discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, lossparams)
    a_fake = generator_B(b, noise) # Fake image generated in domain A
    b_fake = generator_A(a, noise) # Fake image generated in domain B

    a_fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real
    a_real_prob = discriminator_A(a) # Probability that an original image in domain A is real
    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    b_real_prob = discriminator_B(b) # Probability that an original image in domain B is real

    real_A_loss = mean((a_real_prob .- 1) .^ 2)
    fake_A_loss = mean((a_fake_prob .- 0) .^ 2)
    real_B_loss = mean((b_real_prob .- 1) .^ 2)
    fake_B_loss = mean((b_fake_prob .- 0) .^ 2)

    return real_A_loss + fake_A_loss + real_B_loss + fake_B_loss
end
                        
function train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, lossparams)
    # Optimize Discriminators
    ps = params(params(discriminator_A)..., params(discriminator_B)...)
    gs = gradient(() -> discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, lossparams), ps)
    update!(opt_dis, ps, gs)

    # Optimize Generators
    ps = params(params(generator_A)..., params(generator_B)...)
    gs = gradient(() -> generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, lossparams), ps)
    update!(opt_gen, ps, gs)
end

function fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams, dev)
    g_loss, d_loss = 0, 0
    iter = ProgressBar(data.training)
    for epoch in 1:hparams.train.nepochs
        # Training loop
        for (a, b, noise) in iter
            a, b, noise = (FT.(a), FT.(b), FT.(noise)) |> dev
            train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams.loss)
            set_multiline_postfix(iter, "Epoch $epoch\nGenerator Loss: $g_loss\nDiscriminator Loss: $d_loss")
        end

        # error analysis loop
        g_loss, d_loss, count = 0, 0, 0
        for (a, b, noise) = data.validation
            a, b, noise = (FT.(a), FT.(b), FT.(noise)) |> dev
            g_loss += generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams.loss)
            d_loss += discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams.loss)
            count += 1
        end
        g_loss, d_loss = (g_loss, d_loss) ./ count

        if PARGS["logging"]
            @info "metrics" generator_loss = g_loss discriminator_loss = d_loss
            a, b, noise = first(data.validation)
            a, b, noise = (FT.(a), FT.(b), FT.(noise)) |> dev
            b_fake = generator_A(a, noise)
            a_fake = generator_B(b, noise)
            b_rec = generator_A(a_fake, noise)
            a_rec = generator_B(b_fake, noise)
            a, b, a_fake, b_fake, a_rec, b_rec = (a, b, a_fake, b_fake, a_rec, b_rec) |> cpu
            Wandb.log(
                lg, 
                Dict(
                    "real_low_res" => Wandb.Image(a[:, :, 1, 1]),
                    "real_high_res" => Wandb.Image(b[:, :, 1, 1]),
                    "fake_low_res" => Wandb.Image(a_fake[:, :, 1, 1]),
                    "fake_high_res" => Wandb.Image(b_fake[:, :, 1, 1]),
                    "rec_low_res" => Wandb.Image(a_rec[:, :, 1, 1]),
                    "rec_high_res" => Wandb.Image(b_rec[:, :, 1, 1]),
                )
            )
        end

        # checkpointing
        checkpoint_path = joinpath(OUTPUT_DIR, "checkpoint_cyclegan.bson")
        model = (generator_A, generator_B, discriminator_A, discriminator_B) |> cpu
        @save checkpoint_path model opt_gen opt_dis hparams
    end
end

"""
    train(dataset=Turbulence2D(), hparams = HyperParams{FT}(); cuda=true, restart=false)

Train the model either from  scratch (with a fresh set of hyper parameters `hparams`), or
restarted from a previous checkpoint (model+optimizers+hyperparameters).

Note that in the case of a restart, the parameters stored in hparams.optimizer will *not*
be used, as the previously saved optimizers and other hyperparams are read in and used.
"""
function train(dataset=Turbulence2D(), trainparams = TrainingParams(), optimizerparams = OptimizerHyperParams{FT}(), lossparams = LossHyperParams{FT}();cuda=true, restart=true)
    # run with GPU if available
    if cuda && CUDA.has_cuda()
        dev = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        dev = cpu
        @info "Training on CPU"
    end

    # make models
    nchannels = 1
    if restart && isfile(RESTARTFILE)
        @info "Initializing with existing model, optimizers, and hyperparameters."
        @info "Overwriting passed hyperparameters with checkpoint values."
        # First we need to make the model structure
        generator_A = NoisyUNetGenerator2D(nchannels) # Generator For A->B
        generator_B = NoisyUNetGenerator2D(nchannels) # Generator For B->A
        discriminator_A = PatchDiscriminator2D(nchannels) # Discriminator For Domain A
        discriminator_B = PatchDiscriminator2D(nchannels) # Discriminator For Domain B

        # Now load the existing model parameters and fill in the model parameters of the models we just made
        # This also loads the optimizers.
        @load RESTARTFILE model opt_gen opt_dis hparams
        
        loadmodel!(generator_A, model[1])
        loadmodel!(generator_B, model[2])
        loadmodel!(discriminator_A, model[3])
        loadmodel!(discriminator_B, model[4])

        # Push to device
        generator_A = generator_A |> dev
        generator_B = generator_B |> dev
        discriminator_A = discriminator_A |> dev
        discriminator_B = discriminator_B |> dev


    else
        @info "Initializing a new model and optimizers from scratch, using passed hyperparameters."
        generator_A = NoisyUNetGenerator2D(nchannels) |> dev # Generator For A->B
        generator_B = NoisyUNetGenerator2D(nchannels) |> dev # Generator For B->A
        discriminator_A = PatchDiscriminator2D(nchannels) |> dev # Discriminator For Domain A
        discriminator_B = PatchDiscriminator2D(nchannels) |> dev # Discriminator For Domain B
        
        opt_gen = create_optimizer(optimizerparams)
        opt_dis =  create_optimizer(optimizerparams)

        hparams = (; :train => trainparams, :loss => lossparams, :optimizer => optimizerparams)
    end
    # training data
    data = get_dataloader(dataset, batch_size=hparams.train.batch_size)

    if PARGS["logging"]
        update_config!(
            lg, 
            Dict(propertynames(hparams.loss)... .=> getfield.(Ref(hparams.loss), propertynames(hparams.loss))...,
                 propertynames(hparams.optimizer)... .=> getfield.(Ref(hparams.optimizer), propertynames(hparams.optimizer))...,
                 propertynames(hparams.train)... .=> getfield.(Ref(hparams.train), propertynames(hparams.train))...)
               )
    end
    fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams, dev)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

if PARGS["logging"]
    close(lg)
end
