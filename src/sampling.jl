"""
Sample from a diffusion model using the Euler-Maruyama method.

# References
https://arxiv.org/abs/1505.04597
"""
function Euler_Maruyama_sampler(model::CliMAgen.AbstractDiffusionModel, init_x, time_steps, Δt; denoise = false)
    x = mean_x = init_x

    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)
        score = CliMAgen.score(model, x, batch_time_step)

        mean_x = x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt
        x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
    end
    if denoise
        t_end = fill!(similar(init_x, size(init_x)[end]), 1) .* time_steps[end]
        _, σ_t = marginal_prob(model, x, t_end) # This should take x(0), but it does not affect σ_t
        s_t = score(model, x, t_end)
        return x .+ CliMAgen.expand_dims(σ_t, 3) .^ 2 .* s_t # x + σ_t^2 ∇ log P(x)
    else
       return x
    end
end

"""
Sample from a diffusion model using the Predictor-Corrector method.

# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
"""
function predictor_corrector_sampler(model::CliMAgen.AbstractDiffusionModel, init_x, time_steps, Δt, snr=0.16f0; denoise = false)
    x = mean_x = init_x

    @showprogress "Predictor Corrector Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step

        # Corrector step (Langevin MCMC)
        grad = CliMAgen.score(model, x, batch_time_step)

        num_pixels = prod(size(grad)[1:end-1])
        grad_batch_vector = reshape(grad, (size(grad)[end], num_pixels))
        grad_norm = Statistics.mean(sqrt, sum(abs2, grad_batch_vector, dims=2))
        noise_norm = Float32(sqrt(num_pixels))
        langevin_step_size = 2 * (snr * noise_norm / grad_norm)^2
        x += (
            langevin_step_size .* grad .+
            sqrt(2 * langevin_step_size) .* randn!(similar(x))
        )
        # Predictor step (Euler-Maruyama)
        g = CliMAgen.diffusion(model, batch_time_step)
        grad = CliMAgen.score(model, x, batch_time_step)

        mean_x = x .+ CliMAgen.expand_dims((g .^ 2), 3) .* grad .* Δt
        x = mean_x + sqrt.(CliMAgen.expand_dims((g .^ 2), 3) .* Δt) .* randn!(similar(x))
    end
    if denoise
        t_end = fill!(similar(init_x, size(init_x)[end]), 1) .* time_steps[end]
        _, σ_t = marginal_prob(model, x, t_end) # This should take x(0), but it does not affect σ_t
        s_t = score(model, x, t_end)
        return x .+ CliMAgen.expand_dims(σ_t, 3) .^ 2 .* s_t # x + σ_t^2 ∇ log P(x)
    else
       return x
    end
end

"""
Helper function that generates inputs to a sampler.
"""
function setup_sampler(model::CliMAgen.AbstractDiffusionModel, device, tilesize, inchannels; num_images=5, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, num_images) |> device
    init_z = randn(Float32, (tilesize, tilesize, inchannels, num_images)) |> device
    _, σ_T = CliMAgen.marginal_prob(model, zero(init_z), t)
    init_x = (σ_T .* init_z)
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    
    return time_steps, Δt, init_x
end
