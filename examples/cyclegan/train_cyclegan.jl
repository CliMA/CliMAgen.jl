using BSON: @save, @load
using CUDA
using Dates
using Flux
using Flux: params, update!
using FluxTraining
using HDF5
using MLUtils
using ProgressBars
using Statistics: mean
using Wandb
using Logging

using Downscaling
using Downscaling: PatchDiscriminator, UNetGenerator

# dir handling & includes
DATA_DIR = joinpath(pkgdir(Downscaling), "data")
EXAMPLES_DIR = joinpath(pkgdir(Downscaling), "examples")
EXPERIMENT_DIR = joinpath(EXAMPLES_DIR, "cyclegan_cnn")
include(joinpath(EXPERIMENT_DIR, "utils.jl"))
include(joinpath(EXAMPLES_DIR, "artifact_utils.jl"))

# logging
LOGGER = WandbLogger(
        project="cyclegan", 
        name="cyclegan-cnn-$(now())", 
        config = Dict(),
    )
global_logger(LOGGER)


Base.@kwdef struct HyperParams{FT}
    lr::FT = 2e-4
    nepochs::Int = 100
    batch_size::Int = 1
    λ::FT = 10
    λid::FT = 5
end

function generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, hparams)
    a_fake = generator_B(b) # Fake image generated in domain A
    b_fake = generator_A(a) # Fake image generated in domain B

    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    a_fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real

    gen_A_loss = mean((a_fake_prob .- 1) .^ 2)
    rec_A_loss = mean(abs.(b - generator_A(a_fake))) # Cycle-consistency loss for domain B
    idt_A_loss = mean(abs.(generator_A(b) .- b)) # Identity loss for domain B
    gen_B_loss = mean((b_fake_prob .- 1) .^ 2)
    rec_B_loss = mean(abs.(a - generator_B(b_fake))) # Cycle-consistency loss for domain A
    idt_B_loss = mean(abs.(generator_B(a) .- a)) # Identity loss for domain A

    return gen_A_loss + gen_B_loss + hparams.λ * (rec_A_loss + rec_B_loss) + hparams.λid * (idt_A_loss + idt_B_loss)
end

function discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b)
    a_fake = generator_B(b) # Fake image generated in domain A
    b_fake = generator_A(a) # Fake image generated in domain B

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

function train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, hparams)
    # Optimize Discriminators
    ps = params(params(discriminator_A)..., params(discriminator_B)...)
    gs = gradient(() -> discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b), ps)
    update!(opt_dis, ps, gs)

    # Optimize Generators
    ps = params(params(generator_A)..., params(generator_B)...)
    gs = gradient(() -> generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, hparams), ps)
    update!(opt_gen, ps, gs)
end

function fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams)
    # Training loop
    g_loss, d_loss = 0, 0
    iter = ProgressBar(data)
    for epoch in 1:hparams.nepochs
        for (a, b) in iter
            a, b = (a, b) |> devß
            train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, hparams)
            set_multiline_postfix(iter, "Epoch $epoch\nGenerator Loss: $g_loss\nDiscriminator Loss: $d_loss")
        end

        # print current error estimates
        a, b = first(data)
        g_loss = generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, hparams)
        d_loss = discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b,)

        # wandb
        log(
            LOGGER,
            Dict(
                "Generator loss" => g_loss,
                "Discriminator loss" => d_loss,
            ),
        )

        # store current model
        output_path = joinpath(@__DIR__, "output/checkpoint_latest.bson")
        model = (generator_A, generator_B, discriminator_A, discriminator_B) |> cpu
        @save output_path model
    end
end

function train(; cuda=true)
    if cuda && CUDA.has_cuda()
        dev = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        dev = cpu
        @info "Training on CPU"
    end

    # load data setup
    local_dataset_directory = obtain_local_dataset_path(DATA_DIR, moist2d.dataname, moist2d.url, moist2d.filename)
    local_dataset_path = joinpath(local_dataset_directory, moist2d.filename)
    field = "moisture"

    # set hyperparams
    hparams = HyperParams{Float32}()
    update_config!(
        LOGGER, 
        Dict(
            propertynames(hparams) .=> getfield.(Ref(hparams), propertynames(hparams))
        )
    )

    # training data
    data = get_dataloader(local_dataset_path, field=field, split_ratio=0.5, batch_size=hparams.batch_size, dev=dev).training

    # models 
    nchannels = 1
    generator_A = UNetGenerator(nchannels) |> dev # Generator For A->B
    generator_B = UNetGenerator(nchannels) |> dev # Generator For B->A
    discriminator_A = PatchDiscriminator(nchannels) |> dev # Discriminator For Domain A
    discriminator_B = PatchDiscriminator(nchannels) |> dev # Discriminator For Domain B

    # optimizers
    opt_gen = ADAM(hparams.lr, (0.5, 0.999))
    opt_dis = ADAM(hparams.lr, (0.5, 0.999))

    fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams)
end

# run if file is called directly but not if just included
if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

close(LOGGER)
