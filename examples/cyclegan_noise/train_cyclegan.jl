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

using Downscaling
using Downscaling: PatchDiscriminator, UNetGenerator

examples_dir = joinpath(pkgdir(Downscaling), "examples")
include("utils.jl")
include(joinpath(examples_dir, "artifact_utils.jl"))

# Parameters
Base.@kwdef struct HyperParams{FT}
    λ = FT(10.0)
    λid = FT(5.0)
    lr = FT(0.0002)
    nepochs = 100
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

function discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise)
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
    gs = gradient(() -> discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise), ps)
    update!(opt_dis, ps, gs)

    # Optimize Generators
    ps = params(params(generator_A)..., params(generator_B)...)
    gs = gradient(() -> generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams), ps)
    update!(opt_gen, ps, gs)
end

function fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams)
    # Training loop
    g_loss, d_loss = 0, 0
    iter = ProgressBar(data) 
    for epoch in 1:hparams.nepochs
        for (a, b, noise) in iter
            train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
            set_multiline_postfix(iter, "Epoch $epoch\nGenerator Loss: $g_loss\nDiscriminator Loss: $d_loss")
        end

        # print current error estimates
        a, b, noise = first(data)
        g_loss = generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
        d_loss = discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b,noise)

        # store current model
        output_path = joinpath(@__DIR__, "output/checkpoint_latest.bson")
        model = (generator_A, generator_B, discriminator_A, discriminator_B) |> cpu
        @save output_path model
    end
end

function train(path, field, hparams; cuda=true)
    if cuda && CUDA.has_cuda()
        dev = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        dev = cpu
        @info "Training on CPU"
    end

    # training data
    data = get_dataloader(path, field=field, split_ratio=0.5, batch_size=1, dev=dev).training

    # models 
    nchannels = 1
    generator_A = NoisyUNetGenerator(nchannels) |> dev # Generator For A->B
    generator_B = NoisyUNetGenerator(nchannels) |> dev # Generator For B->A
    discriminator_A = PatchDiscriminator(nchannels) |> dev # Discriminator For Domain A
    discriminator_B = PatchDiscriminator(nchannels) |> dev # Discriminator For Domain B

    # optimizers
    opt_gen = ADAM(hparams.lr, (0.5, 0.999))
    opt_dis = ADAM(hparams.lr, (0.5, 0.999))

    fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams,)
end

# run if file is called directly but not if just included
if abspath(PROGRAM_FILE) == @__FILE__
    url = "https://caltech.box.com/shared/static/7oht5betdza54sjftk93ni0bsldtrptl.hdf5"
    filename = "moist2d_512x512.hdf5"
    dataname = "moist2d_512x512"
    # This downloads the data locally, if it not already present, and obtains the location of the directory holding it.
    local_dataset_directory = obtain_local_dataset_path(examples_dir, dataname, url, filename)
    local_dataset_path = joinpath(local_dataset_directory, filename)
    field = "moisture"
    hparams = HyperParams{Float32}()
    train(local_dataset_path, field, hparams)
end
