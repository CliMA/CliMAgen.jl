using Base.Iterators: partition
using Images
using Statistics
using Flux
using Flux: params, update!
using Flux.Data: DataLoader
using Zygote
using Random
using Downscaling: PatchDiscriminator, UNetGenerator
using BSON: @save
using Distributions: Normal, Uniform

function load_image(path, img_size=256, FT=Float32, color_format=RGB)
    # channelview to convert to 3 channels
    img = path |> load .|> color_format |> channelview .|> FT
    img = permutedims(img, (3, 2, 1))

    return imresize(img, (img_size, img_size))
end

function load_dataset(path, img_size=256, FT=Float32, color_format=RGB)
    loader = path -> load_image(path, img_size, FT, color_format)
    imgs = map(loader, path .* readdir(path))
    num_channels = size(imgs[1])[3]

    return reshape(hcat(imgs...), img_size, img_size, num_channels, length(imgs))
end

# Parameters
FT = Float32
device = gpu
num_epochs = 100
batch_size = 4
num_examples = 1000 # Temporary for experimentation
verbose_freq = 10 # Verbose output after every 2 epochs
input_channels = 3
img_size = 28
dis_lr = FT(0.0001)
gen_lr = FT(0.0001)
λ₁ = FT(10.0) # Cycle loss weight for domain A
λ₂ = FT(10.0) # Cycle loss weight for domain B
λid = FT(0.0) # Identity loss weight - Set this to '0' if identity loss is not required
println("Parameter setup: Done!")

# Data Loading
data_folder = "../data/"
experiment_name = "horse2zebra"
trainA = load_dataset(data_folder * experiment_name * "/trainA/", img_size, FT)[:, :, :, 1:num_examples] |> device
trainB = load_dataset(data_folder * experiment_name * "/trainB/", img_size, FT)[:, :, :, 1:num_examples] |> device
train = DataLoader((trainA, trainB), batchsize=batch_size, shuffle=true)
println("Data wrangling: Done!")

# Define models
generator_A = UNetGenerator(input_channels) |> gpu # Generator For A->B
generator_B = UNetGenerator(input_channels) |> gpu # Generator For B->A
discriminator_A = PatchDiscriminator(input_channels) |> gpu # Discriminator For Domain A
discriminator_B = PatchDiscriminator(input_channels) |> gpu # Discriminator For Domain B
println("Model setup: Done!")

# Define Optimizers
opt_gen = ADAM(gen_lr, (0.5, 0.999))
opt_disc = ADAM(dis_lr, (0.5, 0.999))
# opt_disc_A = ADAM(dis_lr, (0.5, 0.999))
# opt_disc_B = ADAM(dis_lr, (0.5, 0.999))
println("Optimizer setup: Done!")

function discriminator_loss(a, b)
    fake_A = generator_B(b) # Fake image generated in domain A
    fake_A_prob = discriminator_A(fake_A) # Probability that generated image in domain A is real
    real_A_prob = discriminator_A(a) # Probability that an original image in domain A is real
    dis_A_real_loss = mean((real_A_prob .- 1) .^ 2)
    dis_A_fake_loss = mean((fake_A_prob .- 0) .^ 2)

    fake_B = generator_A(a) # Fake image generated in domain B
    fake_B_prob = discriminator_B(fake_B) # Probability that generated image in domain B is real
    real_B_prob = discriminator_B(b) # Probability that an original image in domain B is real
    dis_B_real_loss = mean((real_B_prob .- 1) .^ 2)
    dis_B_fake_loss = mean((fake_B_prob .- 0) .^ 2)

    return dis_A_real_loss + dis_A_fake_loss + dis_B_real_loss + dis_B_fake_loss
end

function generator_loss(a, b)
    # Forward Propogation
    fake_B = generator_A(a) # Fake image generated in domain B
    fake_B_prob = discriminator_B(fake_B) # Probability that generated image in domain B is real
    fake_A = generator_B(b) # Fake image generated in domain A
    fake_A_prob = discriminator_A(fake_A) # Probability that generated image in domain A is real

    # Generator loss for domain A->B 
    gen_B_loss = mean((fake_B_prob .- 1) .^ 2)
    rec_B_loss = mean(abs.(b - generator_A(fake_A))) # Cycle-consistency loss for domain B

    # Generator loss for domain B->A 
    gen_A_loss = mean((fake_A_prob .- 1) .^ 2)
    rec_A_loss = mean(abs.(a - generator_B(fake_B))) # Cycle-consistency loss for domain A

    # generator_A (generator_B) should be identity if b (a) is fed : ||gen_A(b) - b|| (||gen_B(a) - a||)
    idt_A_loss = mean(abs.(generator_A(b) - b)) # mae(generator_A(b), b)
    idt_B_loss = mean(abs.(generator_B(a) - a)) # mae(generator_B(a), a)

    return gen_A_loss + gen_B_loss + λ₁ * rec_A_loss + λ₂ * rec_B_loss + λid * (λ₁ * idt_A_loss + λ₂ * idt_B_loss)
end

# Forward prop, backprop, optimise!
function train_step(a, b)
    # Optimize Discriminators
    ps = params(params(discriminator_A)..., params(discriminator_B))
    gs = gradient(() -> discriminator_loss(a, b), ps)
    update!(opt_disc, ps, gs)

    # Optimize Generators
    ps = params(params(generator_A)..., params(generator_B))
    gs = gradient(() -> generator_loss(a, b), ps)
    update!(opt_gen, ps, gs)

    # Forward propagate to collect the losses for keeping track of training progress
    gloss = generator_loss(a, b)
    dloss = discriminator_loss(a, b)

    return gloss, dloss
end

# function save_weights(gen,dis)
#     gen_A = gen_A |> cpu
#     gen_B = gen_B |> cpu
#     dis_A = dis_A |> cpu
#     dis_B = dis_B |> cpu
#     @save "../weights/gen_A.bson" gen_A
#     @save "../weights/gen_B.bson" gen_B
#     @save "../weights/dis_A.bson" dis_A
#     @save "../weights/dis_B.bson" dis_B
# end


# for epoch in 1:100
#     for (x, y) in train
#         # training magic on batch
#         @assert size(x) == (28, 28, 3, 4)
#         @assert size(y) == (28, 28, 3, 4)
#     end
# end

function training()
    println("Training...")
    for epoch in 1:num_epochs
        println("-----------Epoch : $epoch-----------")
        for (a, b) in train
            gloss, dloss = train_step(a, b)
            if epoch % verbose_freq == 0
                println("Generator Loss : $gloss")
                println("Discriminator Loss : $dloss")
            end
        end
    end
    #save_weights()
end

# ### SAMPLING ###
# function sampleA2B(X_A_test)
#     """
#     Samples new images in domain B
#     X_A_test : N x C x H x W array - Test images in domain A
#     """
#     testmode!(gen_A)
#     X_A_test = norm(X_A_test)
#     X_B_generated = cpu(denorm(gen_A(X_A_test |> gpu)).data)
#     testmode!(gen_A,false)
#     imgs = []
#     s = size(X_B_generated)
#     for i in size(X_B_generated)[end]
#        push!(imgs,colorview(RGB,reshape(X_B_generated[:,:,:,i],3,s[1],s[2])))
#     end
#     imgs
# end

# function test()
#    # load test data
#    dataA = load_dataset("../data/trainA/",256)[:,:,:,1:2] |> gpu
#    out = sampleA2B(dataA)
#    for (i,img) in enumerate(out)
#         save("../sample/A_$i.png",img)
#    end
# end

# train()