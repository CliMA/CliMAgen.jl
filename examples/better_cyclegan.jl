using Base.Iterators: partition
using BSON: @save, @load
using Dates
using Statistics: mean
using Images
using Flux
using Flux: params, update!
using Flux.Data: DataLoader
using Random
using Downscaling: PatchDiscriminator, UNetGenerator

# Parameters
FT = Float32
exp_name = "horse2zebra"
input_path = "../data/"
output_path = "./output/"
eval_freq = 1
checkpoint_freq = eval_freq
device = gpu
num_examples = 128
num_epochs = 2
batch_size = 4
img_size = 28
input_channels = 3
dis_lr = FT(0.0001)
gen_lr = FT(0.0001)
λ = FT(10.0)
λid = FT(0.0)

# Define models
generator_A = UNetGenerator(input_channels) |> device # Generator For A->B
generator_B = UNetGenerator(input_channels) |> device # Generator For B->A
discriminator_A = PatchDiscriminator(input_channels) |> device # Discriminator For Domain A
discriminator_B = PatchDiscriminator(input_channels) |> device # Discriminator For Domain B
networks = (generator_A, generator_B, discriminator_A, discriminator_B)

function generator_A_loss(a, b)
    a_fake = generator_B(b) # Fake image generated in domain A
    b_fake = generator_A(a) # Fake image generated in domain B
    fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    gen_loss = mean((fake_prob .- 1) .^ 2)
    rec_loss = mean(abs.(b - generator_A(a_fake))) # Cycle-consistency loss for domain B
    idt_loss = mean(abs.(generator_A(b) - b))

    return gen_loss + λ * rec_loss + λid * λ * idt_loss
end

function generator_B_loss(a, b)
    a_fake = generator_B(b) # Fake image generated in domain A
    b_fake = generator_A(a) # Fake image generated in domain B
    fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real
    gen_loss = mean((fake_prob .- 1) .^ 2)
    rec_loss = mean(abs.(a - generator_B(b_fake))) # Cycle-consistency loss for domain A
    idt_loss = mean(abs.(generator_B(a) - a))

    return gen_loss + λ * rec_loss + λid * λ * idt_loss
end

function discriminator_A_loss(a, b)
    a_fake = generator_B(b) # Fake image generated in domain A
    fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real
    real_prob = discriminator_A(a) # Probability that an original image in domain A is real
    real_loss = mean((real_prob .- 1) .^ 2)
    fake_loss = mean((fake_prob .- 0) .^ 2)

    return real_loss + fake_loss
end

function discriminator_B_loss(a, b)
    b_fake = generator_A(a) # Fake image generated in domain B
    fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    real_prob = discriminator_B(b) # Probability that an original image in domain B is real
    real_loss = mean((real_prob .- 1) .^ 2)
    fake_loss = mean((fake_prob .- 0) .^ 2)

    return real_loss + fake_loss
end

function train_step(opt_gen_A, opt_gen_B, opt_disc_A, opt_disc_B, a, b)
    # Optimize Discriminators
    ps = params(discriminator_A)
    gs = gradient(() -> discriminator_A_loss(a, b), ps)
    update!(opt_disc_A, ps, gs)

    ps = params(discriminator_B)
    gs = gradient(() -> discriminator_B_loss(a, b), ps)
    update!(opt_disc_B, ps, gs)

    # Optimize Generators
    ps = params(generator_A)
    gs = gradient(() -> generator_A_loss(a, b), ps)
    update!(opt_gen_A, ps, gs)

    ps = params(generator_B)
    gs = gradient(() -> generator_B_loss(a, b), ps)
    update!(opt_gen_B, ps, gs)

    # Forward propagate to collect the losses for keeping track of training progress
    gA_loss = generator_A_loss(a, b)
    gB_loss = generator_B_loss(a, b)
    dA_loss = discriminator_A_loss(a, b)
    dB_loss = discriminator_B_loss(a, b)

    return gA_loss, gB_loss, dA_loss, dB_loss
end

function training()
    # Load data
    dataA = load_dataset(input_path * exp_name * "/trainA/", img_size, FT)[:, :, :, 1:num_examples] |> device
    dataB = load_dataset(input_path * exp_name * "/trainB/", img_size, FT)[:, :, :, 1:num_examples] |> device
    data = DataLoader((dataA, dataB), batchsize=batch_size, shuffle=true)

    # Define Optimizers
    opt_gen_A = ADAM(gen_lr, (0.5, 0.999))
    opt_gen_B = ADAM(gen_lr, (0.5, 0.999))
    opt_disc_A = ADAM(dis_lr, (0.5, 0.999))
    opt_disc_B = ADAM(dis_lr, (0.5, 0.999))

    # Training loop
    total_iters = 0
    @info "Training begins..."
    for epoch in 1:num_epochs
        epoch_start = Dates.now()
        @info "Epoch: $epoch -------------------------------------------------------------------"
        for (batch_idx, (a, b)) in enumerate(data)
            a, b = normalize(a), normalize(b)
            gA_loss, gB_loss, dA_loss, dB_loss = train_step(opt_gen_A, opt_gen_B, opt_disc_A, opt_disc_B, a, b)
            total_loss = gA_loss + gB_loss + dA_loss + dB_loss
            total_iters += batch_size

            if total_iters % eval_freq == 0
                @info "Total iteration: $total_iters - Generator losses: $gA_loss, $gB_loss, Discriminator losses: $dA_loss, $dB_loss"
            end

            if total_iters % checkpoint_freq == 0
                @info "Total iteration: $total_iters - Checkpointing model."
                file_iter = output_path * exp_name * "/checkpoint_iteration_$(total_iters)_loss_$(total_loss).bson"
                file_last = output_path * exp_name * "/checkpoint_latest.bson"
                networks_cpu = networks |> cpu
                @save file_iter networks_cpu
                cp(file_iter, file_last, force=true)
            
                prefix_path = output_path * exp_name * "/training/" * "image_$(total_iters)"
                save_model_samples(prefix_path, a, b)
            end
        end
        @info "Epoch duration: $(Dates.canonicalize(Dates.now() - epoch_start))"
    end
end

function testing()
    # Load data
    dataA = load_dataset(input_path * exp_name * "/testA/", img_size, FT)[:, :, :, 1:num_examples] |> device
    dataB = load_dataset(input_path * exp_name * "/testB/", img_size, FT)[:, :, :, 1:num_examples] |> device
    data = DataLoader((dataA, dataB), batchsize=1, shuffle=false)

    # Load generators
    genA, genB, _, _ = @load output_path * exp_name * "checkpint_latest.bson"
    Flux.loadmodel!(generator_A, genA)
    Flux.loadmodel!(generator_B, genB)

    # Testing loop
    @info "Testing begins..."
    for (sample_idx, (a, b)) in enumerate(data)
        a, b = normalize(a), normalize(b)
        prefix_path = output_path * exp_name * "/testing/" * "image_$(sample_idx)"
        save_model_samples(path, a, b)
    end
    @info "Testing complete."
end

# Utils
function normalize(x)
    return @. 2 * x - 1
end

function unnormalize(x)
    return @. (x + 1) / 2
end

function load_image(path, img_size=256, FT=Float32, color_format=RGB)
    # channelview to convert to 3 channels
    img = path |> load .|> color_format |> channelview .|> FT
    img = permutedims(img, (3, 2, 1))

    return imresize(img, (img_size, img_size))
end

function save_image(path, img, color_format=RGB)
    img = permutedims(img, (3, 2, 1))
    img = colorview(color_format, img)
    save(path, img)
end

function load_dataset(path, img_size=256, FT=Float32, color_format=RGB)
    loader = path -> load_image(path, img_size, FT, color_format)
    imgs = map(loader, path .* readdir(path))

    return Flux.stack(imgs, dims=4)
end

function save_model_samples(prefix_path, a, b)
    fake_B = generator_A(a) |> cpu
    fake_A = generator_B(b) |> cpu
    rec_B = generator_A(fake_A) |> cpu
    rec_A = generator_B(fake_B) |> cpu

    # Save the images
    imgs = map(unnormalize, [a, fake_A, rec_A, b, fake_B, rec_B])
    suffices = ["real_A", "fake_A", "rec_A", "real_B", "fake_B", "rec_B"]
    for (img_batch, suffix) in zip(imgs, suffices)
        for img_idx in 1:size(img_batch)[end]
            path = prefix_path * "_$(suffix)_$(img_idx).png"
            save_image(path, img_batch[:, :, :, img_idx])
        end
    end
end
