using ArgParse
using BSON: @save, @load
using CUDA
using Dates
using Flux
using Flux: params, update!, loadmodel!
using FluxTraining
using HDF5
using MLUtils
using ProgressBars
using Statistics: mean

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
EXAMPLE_NAME = "turbulence_2d/dcgan"
TIME = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")

# set up storage directory for experiment
OUTPUT_DIR = joinpath(OUTDIR, EXAMPLE_NAME, TIME)
mkpath(OUTPUT_DIR)

# get access to the datasets and dataloaders
include(joinpath(DATADIR, "utils_data.jl"))

# fix float type
const FT = Float32


Base.@kwdef struct HyperParams{FT}
    lr = FT(0.0002)
    nepochs::Int = 100
    batch_size::Int = 1
end

function generator_loss(generator_A, discriminator_B, a, b, hparams)
    b_fake = generator_A(a) # Fake image generated in domain A
    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    gen_B_loss = mean((b_fake_prob .- 1) .^ 2)

    return gen_B_loss
end

function discriminator_loss(generator_A, discriminator_B, a, b, hparams)
    b_fake = generator_A(a) # Fake image generated in domain B

    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    b_real_prob = discriminator_B(b) # Probability that an original image in domain B is real

    real_B_loss = mean((b_real_prob .- 1) .^ 2)
    fake_B_loss = mean((b_fake_prob .- 0) .^ 2)

    return real_B_loss + fake_B_loss
end

function train_step!(opt_gen, opt_dis, generator_A, discriminator_B, a, b, hparams)
    # Optimize Discriminators
    ps = params(discriminator_B)
    gs = gradient(() -> discriminator_loss(generator_A, discriminator_B, a, b, hparams), ps)
    update!(opt_dis, ps, gs)

    # Optimize Generators
    ps = params(generator_A)
    gs = gradient(() -> generator_loss(generator_A, discriminator_B, a, b, hparams), ps)
    update!(opt_gen, ps, gs)
end

function fit!(opt_gen, opt_dis, generator_A, discriminator_B, data, hparams, dev)
    g_loss, d_loss = 0, 0
    iter = ProgressBar(data.training)
    for epoch in 1:hparams.nepochs
        # training lopp
        for (_, b, _) in iter
            a, b = (2 .* rand(FT, size(b)...) .- 1, FT.(b)) .|> dev
            train_step!(opt_gen, opt_dis, generator_A, discriminator_B, a, b, hparams)
            set_multiline_postfix(iter, "Epoch $epoch\nGenerator Loss: $g_loss\nDiscriminator Loss: $d_loss")
        end

        # error analysis loop
        g_loss, d_loss, count = 0, 0, 0
        for (_, b, _) = data.validation
            a, b = (2 .* rand(FT, size(b)...) .- 1, FT.(b)) |> dev
            g_loss += generator_loss(generator_A, discriminator_B, a, b, hparams)
            d_loss += discriminator_loss(generator_A, discriminator_B, a, b, hparams)
            count += 1
        end
        g_loss, d_loss = (g_loss, d_loss) ./ count

        # checkpointing
        checkpoint_path = joinpath(output_dir, "checkpoint_dcgan.bson")
        model = (generator_A, discriminator_B) |> cpu
        @save checkpoint_path model opt_gen opt_dis
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

    # make models
    nchannels = 1
    if restart && isfile(output_filepath)
        @info "Initializing with existing model and optimizers"

        # First we need to make the model structure
        generator_A = UNetGenerator2D(nchannels) |> dev
        discriminator_B = PatchDiscriminator2D(nchannels) |> dev

        # Now load the existing model parameters and fill in the parameters of the models we just made
        # This also loads the optimizers
        @load output_filepath model opt_gen opt_dis
        loadmodel!(generator_A, model[1])
        loadmodel!(discriminator_B, model[2])

        # Push to device
        generator_A = generator_A |> dev
        discriminator_B = discriminator_B |> dev
    else
        @info "Initializing a new model and optimizers from scratch"
        generator_A = UNetGenerator2D(nchannels) |> dev
        discriminator_B = PatchDiscriminator2D(nchannels) |> dev

        # optimizers
        opt_gen = ADAM(hparams.lr, (0.5, 0.999))
        opt_dis = ADAM(hparams.lr, (0.5, 0.999))
    end

    fit!(opt_gen, opt_dis, generator_A, discriminator_B, data, hparams, dev)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
