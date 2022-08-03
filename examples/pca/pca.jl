using Base.Iterators: partition
using BSON: @save, @load
using Dates
using Statistics: mean
using Images

using Flux
using Flux: params, update!
using Flux.Data: DataLoader
using Random

using Parameters: @with_kw
using CUDA
using FFTW
using MultivariateStats
using MultivariateStats: fit, predict
using UnicodePlots
using Printf

CUDA.allowscalar(false)

# Parameters
@with_kw struct HyperParams
    exp_name = "moist2d"
    input_path = "../../data/"
    output_path = "./output/"
    eval_freq = 700
    checkpoint_freq = eval_freq
    device = gpu
    num_examples = 700
    num_epochs = 1 # 100
    batch_size = 1
    num_latent_dim = 512
    img_size = 256
    input_channels = 1
    lr_gen = Float32(0.0002)
    lr_dscr = Float32(0.0002)
    λ = Float32(10.0)
    λid = Float32(0.5)
    color_format = Gray
end

function gen_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, λ, λid)
    lores_fake = gen_lores(hires) # Fake image generated in lores domain
    hires_fake = gen_hires(lores) # Fake image generated in hires domain

    hires_fake_prob = dscr_hires(hires_fake) # Probability that generated image in hires domain is real
    lores_fake_prob = dscr_lores(lores_fake) # Probability that generated image in lores domain is real

    gen_lores_loss = mean((lores_fake_prob .- 1) .^ 2)
    rec_lores_loss = mean(abs.(hires - gen_hires(lores_fake))) # Cycle-consistency loss for hires domain
    idt_lores_loss = mean(abs.(gen_hires(hires) .- hires)) # Identity loss for hires domain
    gen_hires_loss = mean((hires_fake_prob .- 1) .^ 2)
    rec_hires_loss = mean(abs.(lores - gen_lores(hires_fake))) # Cycle-consistency loss for lores domain
    idt_hires_loss = mean(abs.(gen_lores(lores) .- lores)) # Identity loss for lores domain

    return gen_lores_loss + gen_hires_loss + λ * (rec_lores_loss + rec_hires_loss + λid * (idt_lores_loss + idt_hires_loss))
end

function dscr_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires)
    lores_fake = gen_lores(hires) # Fake image generated in lores domain
    hires_fake = gen_hires(lores) # Fake image generated in hires domain

    lores_fake_prob = dscr_lores(lores_fake) # Probability that generated image in lores domain is real
    lores_real_prob = dscr_lores(lores) # Probability that an original image in lores domain is real
    hires_fake_prob = dscr_hires(hires_fake) # Probability that generated image in hires domain is real
    hires_real_prob = dscr_hires(hires) # Probability that an original image in hires domain is realhires
    real_lores_loss = mean((lores_real_prob .- 1) .^ 2)
    fake_lores_loss = mean((lores_fake_prob .- 0) .^ 2)
    real_hires_loss = mean((hires_real_prob .- 1) .^ 2)
    fake_hires_loss = mean((hires_fake_prob .- 0) .^ 2)

    return real_lores_loss + fake_lores_loss + real_hires_loss + fake_hires_loss
end

function train_step!(opt_gen, opt_dis, gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams)
    # Optimize Discriminators
    ps = params(params(dscr_lores)..., params(dscr_hires)...)
    gs = gradient(() -> dscr_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires), ps)
    update!(opt_dis, ps, gs)

    # Optimize Generators
    ps = params(params(gen_hires)..., params(gen_lores)...)
    gs = gradient(() -> gen_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams.λ, hparams.λid), ps)
    update!(opt_gen, ps, gs)

    g_loss = gen_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams.λ, hparams.λid)
    d_loss = dscr_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires)

    return g_loss, d_loss
end

function train(; kwargs...)
    # Model Parameters
    hparams = HyperParams(; kwargs...)

    if CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # Load MNIST dataset
    input_path = hparams.input_path
    exp_name = hparams.exp_name
    img_size = hparams.img_size
    color_format = hparams.color_format
    num_examples = hparams.num_examples
    data_lores_raw = load_dataset(input_path * exp_name * "/trainA/", img_size, Float32, color_format)[:, :, :, 1:num_examples]
    data_hires_raw = load_dataset(input_path * exp_name * "/trainB/", img_size, Float32, color_format)[:, :, :, 1:num_examples]

    # Transform data
    μ_lores = mean(data_lores_raw, dims=4)
    σ_lores = std(data_lores_raw, dims=4)
    data_lores = (data_lores_raw .- μ_lores) ./ σ_lores
    data_lores = fft(data_lores, 1:2)
    data_lores = reshape(data_lores, hparams.img_size * hparams.img_size, hparams.num_examples)
    data_lores = vcat(real(data_lores), imag(data_lores))
    pca_lores = fit(PCA, data_lores, maxoutdim=hparams.num_latent_dim)
    data_lores = predict(pca_lores, data_lores)
    μ_fft_lores = mean(data_lores, dims=2)
    σ_fft_lores = std(data_lores, dims=2)
    data_lores = (data_lores .- μ_fft_lores) ./ σ_fft_lores
    @info "Variance retained in low resolution data:"
    @info principalratio(pca_lores)

    μ_hires = mean(data_hires_raw, dims=4)
    σ_hires = std(data_hires_raw, dims=4)
    data_hires = (data_hires_raw .- μ_hires) ./ σ_hires
    data_hires = fft(data_hires, 1:2)
    data_hires = reshape(data_hires, hparams.img_size * hparams.img_size, hparams.num_examples)
    data_hires = vcat(real(data_hires), imag(data_hires))
    pca_hires = fit(PCA, data_hires, maxoutdim=hparams.num_latent_dim)
    data_hires = predict(pca_hires, data_hires)
    μ_fft_hires = mean(data_hires, dims=2)
    σ_fft_hires = std(data_hires, dims=2)
    data_hires = (data_hires .- μ_fft_hires) ./ σ_fft_hires
    @info "Variance retained in high resolution data:"
    @info principalratio(pca_hires)

    # Partition into batches
    data = DataLoader((data_lores, data_hires), batchsize=hparams.batch_size, shuffle=true)

    # # Instantiate the models
    # dscr_hires = Discriminator() |> device
    # dscr_lores = Discriminator() |> device
    # gen_hires = Generator(hparams.latent_dim) |> device
    # gen_lores = Generator(hparams.latent_dim) |> device

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr, (0.5, 0.999))
    opt_gen = ADAM(hparams.lr_gen, (0.5, 0.999))

    # Training loop
    num_examples = 0
    @info "Training begins..."
    for epoch in 1:hparams.num_epochs
        epoch_start = Dates.now()
        @info "Epoch: $epoch -------------------------------------------------------------------"
        for (lores, hires) in data
            lores = lores |> device
            hires = hires |> device
            # g_loss, d_loss = train_step(opt_gen, opt_dscr, gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams)

            # if num_examples % hparams.eval_freq == 0
            #     @info "Training examples: $num_examples - Generator loss: $g_loss, Discriminator loss: $d_loss"
            # end

            if num_examples % hparams.checkpoint_freq == 0
                # Save model state
                @info "Training examples: $num_examples - Checkpointing model."
                # file_last = output_path * exp_name * "/checkpoint_latest.bson"
                # networks_cpu = (gen_lores, gen_hires, dscr_lores, dscr_hires) |> cpu
                # @save file_last networks_cpu
            
                # Generate fake latent images
                lores_sample, hires_sample = lores[:, 1], hires[:, 1]
                # hires_fake = gen_hires(lores_sample) |> cpu
                # lores_fake = gen_lores(hires_sample) |> cpu
                # hires_rec = gen_hires(lores_fake) |> cpu
                # lores_rec = gen_lores(hires_fake) |> cpu
                lores_sample = lores_sample |> cpu
                hires_sample = hires_sample |> cpu
                #lores_series = [lores_sample, lores_fake, lores_rec]
                #hires_series = [hires_sample, hires_fake, hires_rec]
                lores_series = [lores_sample]
                hires_series = [hires_sample]
            
                # Reconstruct full images
                lores_series = map(lores_series) do x
                    offset = hparams.img_size * hparams.img_size
                    x = μ_fft_lores .+ σ_fft_lores .* x
                    x = reconstruct(pca_lores, x)
                    x = reshape(x[1:offset] + im * x[offset+1:end], hparams.img_size, hparams.img_size)
                    x = real(ifft(x))
                    μ_lores .+ σ_lores .* x

                    # TODO: remove below
                    (x .- minimum(x)) ./ (maximum(x) - minimum(x))
                end
            
                hires_series = map(hires_series) do x
                    offset = hparams.img_size * hparams.img_size
                    x = μ_fft_hires .+ σ_fft_hires .* x
                    x = reconstruct(pca_hires, x)
                    x = reshape(x[1:offset] + im * x[offset+1:end], hparams.img_size, hparams.img_size)
                    x = real(ifft(x))
                    x = μ_hires .+ σ_hires .* x

                    # TODO: remove below
                    (x .- minimum(x)) ./ (maximum(x) - minimum(x))
                end
            
                prefix_path = hparams.output_path * hparams.exp_name * "/training/" * "image_$epoch"
                save_reconstructed_images(prefix_path, vcat(lores_series, hires_series))
            end
        
            num_examples += hparams.batch_size
        end 

        @info "Epoch duration: $(Dates.canonicalize(Dates.now() - epoch_start))"
    end
end

function load_image(path, img_size=256, FT=Float32, color_format=RGB)
    # channelview to convert to 3 channels
    img = path |> load .|> color_format |> channelview .|> FT
    if length(size(img)) == 3
        img = permutedims(img, (3, 2, 1))
    end

    return imresize(img, (img_size, img_size))
end

function load_dataset(path, img_size=256, FT=Float32, color_format=RGB)
    loader = path -> load_image(path, img_size, FT, color_format)
    imgs = map(loader, path .* readdir(path))

    return Flux.stack(imgs, dims=4)
end

function save_reconstructed_images(prefix_path, images, color_format=Gray)
    # suffices = ["lores_real", "lores_fake", "lores_rec", "hires_real", "hires_fake", "hires_rec"]
    suffices = ["lores_real", "hires_real"]
    for (img, suffix) in zip(images, suffices)
        path = prefix_path * "_$(suffix).png"
        save_image(path, img[:, :, :, 1], color_format)
    end
end

function save_image(path, img, color_format=RGB)
    if size(img)[3] == 3
        img = permutedims(img, (3, 2, 1))
    elseif size(img)[3] == 1
        img = img[:, :, 1]
    end
    img = colorview(color_format, img)
    save(path, img)
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
