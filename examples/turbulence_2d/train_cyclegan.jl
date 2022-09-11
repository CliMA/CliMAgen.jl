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
using Downscaling: PatchDiscriminator2D, UNetGenerator2D

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


Base.@kwdef struct HyperParams{FT}
    λ = FT(10.0)
    λid = FT(5.0)
    lr = FT(0.0002)
    ϵ = FT(1e-8)
    gradclip = FT(1.0)
    β1 = FT(0.9)
    β2 = FT(0.999)
    warmup::Int = 5000
    nepochs::Int = 100
    batch_size::Int = 1
end



function generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
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

    return gen_A_loss + gen_B_loss + hparams.λ * (rec_A_loss + rec_B_loss) + hparams.λid * (idt_A_loss + idt_B_loss)
end

function discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
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
                        
function train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
    # Optimize Discriminators
    ps = params(params(discriminator_A)..., params(discriminator_B)...)
    gs = gradient(() -> discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams), ps)
    update!(opt_dis, ps, gs)

    # Optimize Generators
    ps = params(params(generator_A)..., params(generator_B)...)
    gs = gradient(() -> generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams), ps)
    update!(opt_gen, ps, gs)
end

function fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams, dev, step)
    g_loss, d_loss = 0, 0
    iter = ProgressBar(data.training)
    for epoch in 1:hparams.nepochs
        # Training loop
        for (a, b, noise) in iter
            a, b, noise = (FT.(a), FT.(b), FT.(noise)) |> dev
            # Apply the warmup schedule to the learning rate, if applicable
            # Note that we could use ParameterSchedulers.jl, but in the end their
            # exampling of simple warmup schedule appeared less readable than the below
            # https://fluxml.ai/ParameterSchedulers.jl/dev/docs/tutorials/warmup-schedules.html
            # and we would still need to enumerate the step
            # https://fluxml.ai/Flux.jl/stable/training/optimisers/#Scheduling-Optimisers

            # We chained together two optimizers in our opt structs.
            # The first is gradient clipping, the second is Adam, where eta, the lr, is stored.
            if hparams.warmup  > 1
                opt_dis[2].eta = hparams.lr*FT(min(1.0, step / hparams.warmup))
                opt_gen[2].eta = hparams.lr*FT(min(1.0, step / hparams.warmup))
            end

            train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
            set_multiline_postfix(iter, "Epoch $epoch\nGenerator Loss: $g_loss\nDiscriminator Loss: $d_loss")
            step += 1
        end

        # error analysis loop
        g_loss, d_loss, count = 0, 0, 0
        for (a, b, noise) = data.validation
            a, b, noise = (FT.(a), FT.(b), FT.(noise)) |> dev
            g_loss += generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
            d_loss += discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
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
        @save checkpoint_path model opt_gen opt_dis step
    end
end

function train(dataset=Turbulence2D(), hparams=HyperParams{FT}(); cuda=true, restart=false)
    # run with GPU if available
    if cuda && CUDA.has_cuda()
        dev = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        dev = cpu
        @info "Training on CPU"
    end

    # training data
    data = get_dataloader(dataset, batch_size=hparams.batch_size)

    if PARGS["logging"]
        update_config!(
            lg, 
            Dict(propertynames(hparams) .=> getfield.(Ref(hparams), propertynames(hparams)))
    )
    end

    # make models
    nchannels = 1
    if restart && isfile(RESTARTFILE)
        @info "Initializing with existing model and optimizers"
        # Possible issue is that hparams are wrapped up in the optimizer, which we read in from the
        # checkpoint. but we do not update this optimizer with e.g. the learning rate or gradclip
        # from the passed value of hparams! in effect, there are hyper params that are not used
        # but are still passed.

        # First we need to make the model structure
        generator_A = NoisyUNetGenerator2D(nchannels) # Generator For A->B
        generator_B = NoisyUNetGenerator2D(nchannels) # Generator For B->A
        discriminator_A = PatchDiscriminator2D(nchannels) # Discriminator For Domain A
        discriminator_B = PatchDiscriminator2D(nchannels) # Discriminator For Domain B

        # Now load the existing model parameters and fill in the parameters of the models we just made
        # This also loads the optimizers
        @load RESTARTFILE model opt_gen opt_dis step
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
        @info "Initializing a new model and optimizers from scratch"
        generator_A = NoisyUNetGenerator2D(nchannels) |> dev # Generator For A->B
        generator_B = NoisyUNetGenerator2D(nchannels) |> dev # Generator For B->A
        discriminator_A = PatchDiscriminator2D(nchannels) |> dev # Discriminator For Domain A
        discriminator_B = PatchDiscriminator2D(nchannels) |> dev # Discriminator For Domain B


        # Per Optimizers: Gradient clipping is an AbstractOptimizer, and optimizers can be composed
        # https://fluxml.ai/Flux.jl/stable/training/optimisers/#Gradient-Clipping
        # https://fluxml.ai/Flux.jl/stable/training/optimisers/#Composing-Optimisers
        # so, we can do that here. Weight decay would also be composed if we include that

        # Note that we can also do the clipping not as part of the optimizer, by applying a clip function that we write
        # prior to updating the optimizer.

        if hparams.gradclip >0
            gradclip = hparams.gradclip
        else
            gradclip  = FT(1.0/eps(FT)) # this would result in no clipping
        end
        opt_gen = Flux.Optimiser(ClipNorm(gradclip), Adam(hparams.lr, (hparams.β1, hparams.β2), hparams.ϵ))
        opt_dis =  Flux.Optimiser(ClipNorm(gradclip),Adam(hparams.lr, (hparams.β1, hparams.β2), hparams.ϵ))

        step = 1
    end
    # `step` stand-in for state, which will include moving average?
    fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams, dev, step)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

if PARGS["logging"]
    close(lg)
end
