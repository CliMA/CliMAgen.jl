using BSON
using Flux: chunk
using Images
using ProgressMeter
using Plots
using Statistics: mean

# our training file
include("train_diffusion.jl")

"""
Sample from a diffusion model using the Euler-Maruyama method.

# References
https://arxiv.org/abs/1505.04597
"""
function Euler_Maruyama_sampler(model::CliMAgen.AbstractDiffusionModel, init_x, time_steps, Δt)
    x = mean_x = init_x

    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)
        score = CliMAgen.score(model, x, batch_time_step)

        mean_x = x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt
        x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn(Float32, size(x))
    end
    return mean_x
end

"""
Sample from a diffusion model using the Predictor-Corrector method.

# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
"""
function predictor_corrector_sampler(model::CliMAgen.AbstractDiffusionModel,  init_x, time_steps, Δt, snr=0.16f0)
    x = mean_x = init_x

    @showprogress "Predictor Corrector Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step

        # Corrector step (Langevin MCMC)
        grad = CliMAgen.score(model, x, batch_time_step)

        num_pixels = prod(size(grad)[1:end-1])
        grad_batch_vector = reshape(grad, (size(grad)[end], num_pixels))
        grad_norm = mean(sqrt, sum(abs2, grad_batch_vector, dims=2))
        noise_norm = Float32(sqrt(num_pixels))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm)^2
        x += (
            langevin_step_size .* grad .+
            sqrt(2 * langevin_step_size) .* randn(Float32, size(x))
        )
        # Predictor step (Euler-Maruyama)
        g = CliMAgen.diffusion(model, batch_time_step)
        grad = CliMAgen.score(model, x, batch_time_step)

        mean_x = x .+ CliMAgen.expand_dims((g .^ 2), 3) .* grad .* Δt
        x = mean_x + sqrt.(CliMAgen.expand_dims((g .^ 2), 3) .* Δt) .* randn(Float32, size(x))
    end
    return mean_x
end

function plot_result(model, save_path, hpdata)
    device = cpu
    @info "Using device: $device"
    model = model |> device
    time_steps, Δt, init_x = setup_sampler(model, device, hpdata)

    # Euler-Maruyama
    euler_maruyama = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
    sampled_noise = convert_to_image(init_x, size(init_x)[end])
    save(joinpath(save_path, "sampled_noise.jpeg"), sampled_noise)
    em_image = MLDatasets.convert2image(MLDatasets.CIFAR10, euler_maruyama)[:,:,1]
    save(joinpath(save_path, "em_images.jpeg"), em_image)

    # Predictor Corrector
    pc = predictor_corrector_sampler(model, init_x, time_steps, Δt)
    pc_image = MLDatasets.convert2image(MLDatasets.CIFAR10, pc)[:,:,1]
    save(joinpath(save_path, "pc_images.jpeg"), pc_image)
end

"""
Helper function that produces images from a batch of images.
"""
function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 32, :)...), (2, 1)))
end

"""
Helper to make an animation from a batch of images.
"""
function convert_to_animation(x)
    frames = size(x)[end]
    batches = size(x)[end-1]
    animation = @animate for i = 1:frames+frames÷4
        if i <= frames
            heatmap(
                convert_to_image(x[:, :, :, :, i], batches),
                title="Iteration: $i out of $frames"
            )
        else
            heatmap(
                convert_to_image(x[:, :, :, :, end], batches),
                title="Iteration: $frames out of $frames"
            )
        end
    end
    return animation
end

"""
Helper function that generates inputs to a sampler.
"""
function setup_sampler(model::CliMAgen.AbstractDiffusionModel, device, hpdata; num_images=5, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, num_images) |> device
    init_z = randn(Float32, (32, 32, hpdata.inchannels, num_images))
    _, σ_T = CliMAgen.marginal_prob(model, zero(init_z), t)
    init_x = (σ_T .* init_z) |> device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

if abspath(PROGRAM_FILE) == @__FILE__
    save_path = "examples/cifar10/output"
    checkpoint_path = joinpath(save_path, "checkpoint_model.bson")
    ############################################################################
    # Issue loading function closures with BSON:
    # https://github.com/JuliaIO/BSON.jl/issues/69
    #
    BSON.@load checkpoint_path model hp
    #
    # BSON.@load does not work if defined inside plot_result(⋅) because
    # it contains a function closure, GaussFourierProject(⋅), containing W.
    ###########################################################################
    plot_result(model, save_path, hp.data)
end
