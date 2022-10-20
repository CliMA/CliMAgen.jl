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
        x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
    end
    return x
end


"""
Sample from a diffusion model using the Predictor-Corrector method.

# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
"""
function predictor_corrector_sampler(model::CliMAgen.AbstractDiffusionModel, init_x, time_steps, Δt, snr=0.16f0)
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
    return x
end

"""
Helper function that generates inputs to a sampler.
"""
function setup_sampler(model::CliMAgen.VarianceExplodingSDE, device, tilesize, inchannels; linear_in_ρ = false, nimages=5, nsteps=500, ϵ=1.0f-5)
    t = ones(Float32, nimages) |> device
    init_z = randn(Float32, (tilesize, tilesize, inchannels, nimages)) |> device
    _, σ_T = CliMAgen.marginal_prob(model, zero(init_z), t)
    init_x = (σ_T .* init_z)
    if !linear_in_ρ
        time_steps = LinRange(1.0f0, ϵ, nsteps)
    else
        σ_max = model.σ_max
        σ_min = model.σ_min
        ρ1 = 2*σ_max
        ρϵ = 2*σ_min*(σ_max/σ_min)^ϵ
        ρ_steps = LinRange(ρ1, ρϵ, nsteps)
        time_steps = log.(ρ_steps ./(2*σ_min) - 1) ./ log(σ_max/σ_min)
    end
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end


"""
Sample from a diffusion model using the Euler-Maruyama method.

# References
https://arxiv.org/abs/1505.04597
"""
function exponential_Euler_Maruyama_sampler(model::CliMAgen.VarianceExplodingSDE, init_x, time_steps, Δt)
    x = mean_x = init_x
    # SDE: dx = -g(t)^2/σ net(x,t)*dt + g(t) dW
    # Only treat the net as fixed over the timestep when discretizing
    @showprogress "exponential Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)
        net = model.net(x, batch_time_step)
        #∫ g(t)^2/σ(t) = 2σ(t)
        #∫ g(t)^2 = σ(t)^2
        σ_max = model.σ_max
        σ_min = model.σ_min
        σ_t = σ_min*(σ_max/σ_min)^time_step #σ(t)
        σ_tnext = σ_min*(σ_max/σ_min)^(time_step - Δt)#σ(t-Δt)
        Δσ = σ_t - σ_tnext
        
        mean_x = x .+ net .* Δσ .* 2
        x = mean_x .+  CliMAgen.expand_dims(sqrt(σ_t^2 - σ_tnext^2), 3) .* randn!(similar(x))
    end
    return x
end