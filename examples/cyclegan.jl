using Base.Iterators: partition
using Images
using Statistics
using Flux
using Flux: gpu, ADAM
using Random
using Printf
using Downscaling: CycleGAN, PatchDiscriminator, UNetGenerator, update_cyclegan

size_img = 28
data_folder = "../data"

Base.@kwdef struct HyperParams
    batch_size::Int = 4
    epochs::Int = 100
    verbose_freq::Int = 100
    size_dataset::Int = 1000
    lr_dscr_A::Float64 = 0.00005
    lr_gen_A::Float64 = 0.00005
    lr_dscr_B::Float64 = 0.00005
    lr_gen_B::Float64 = 0.00005
end

function convertI2Float(img)
    img_resize = imresize(float.(img), (size_img, size_img))
    if length(size(img_resize)) == 2
        img_resize = RGB.(img_resize)
    end
    return permutedims(channelview(img_resize), (3, 2, 1))
end

function load_images(path::String, size::Int)
	images= zeros(Float32,size_img,size_img,3,size)
	for (index, img) in enumerate(readdir(path, join = true))
		images[:,:,:,index] = convertI2Float(load(img))
        if index == size
            break
        end
	end
	return images
end

function load_data(hparams)
    # Load folder dataset
    images_A = load_images(data_folder * "/horse2zebra/trainA/", hparams.size_dataset)
    images_B = load_images(data_folder * "/horse2zebra/trainB/", hparams.size_dataset)
    data = [ (images_A[:,:,:, r], images_B[:,:,:, r]) |> gpu for r in partition(1:hparams.size_dataset, hparams.batch_size+1)]
    return data
end

function create_output_image(gen, image)
    @eval Flux.istraining() = false
    fake_image = cpu(gen(image))
    @eval Flux.istraining() = true
    image_array = permutedims(dropdims(fake_image; dims=4), (3,2,1))
    image_array = colorview(RGB, image_array)
    return clamp01nan.(image_array)
end

function train()
    hparams = HyperParams()

    data = load_data(hparams)

    # test images
    test_images_A = zeros(Float32, size_img, size_img, 3, 1)
    test_images_B = zeros(Float32, size_img, size_img, 3, 1)
    test_images_A[:, :, :, 1] = convertI2Float(load(data_folder * "/horse2zebra/testA/n02381460_1000.jpg"))
    test_images_B[:, :, :, 1] = convertI2Float(load(data_folder * "/horse2zebra/testB/n02391049_100.jpg"))

    # Discriminator
    dscr_A = PatchDiscriminator(3)
    dscr_B = PatchDiscriminator(3)

    # Generator
    gen_A = UNetGenerator(3, 64) |> gpu
    gen_B = UNetGenerator(3, 64) |> gpu

    # Optimizers
    opt_dscr = ADAM(hparams.lr_dscr_A, (0.5, 0.99))
    opt_gen = ADAM(hparams.lr_gen_A, (0.5, 0.99))

    # GAN
    cyclegGAN = CycleGAN(gen_A, gen_B, dscr_A, dscr_B)

    # Training
    isdir("output") || mkdir("output")
    train_steps = 0
    for ep in 1:hparams.epochs
        @info "Epoch $ep"
        for (x, y) in data
            # Update discriminator and generator
            loss = update_cyclegan(opt_gen, opt_dscr, cyclegGAN, x, y)
            if train_steps % hparams.verbose_freq == 0
                @info("Train step $(ep), Discriminator loss = $(loss["D_loss"]), Generator loss = $(loss["G_loss"])")

                # Save generated fake image
                output_image_A = create_output_image(cyclegGAN.generator_AB, test_images_B |> gpu)
                output_image_B = create_output_image(cyclegGAN.generator_BA, test_images_A |> gpu)
                save(@sprintf("output/cgan_A_steps_%06d.png", train_steps), output_image_A)
                save(@sprintf("output/cgan_B_steps_%06d.png", train_steps), output_image_B)
            end
            train_steps += 1
        end
    end
    @info("Finish  training")
    output_image_A = create_output_image(cyclegGAN.generator_AB, test_images_B |> gpu)
    output_image_B = create_output_image(cyclegGAN.generator_BA, test_images_A |> gpu)
    save("output/cgan_A_steps_final.png", output_image_A)
    save("output/cgan_B_steps_final.png", output_image_B)
end

train()
