"""
    Euler_Maruyama_sampler(model::CliMAgen.AbstractDiffusionModel,
                           init_x::A,
                           time_steps,
                           Δt;
                           c=nothing,
                           forward = false
                           )::A where {A}

Generate a sample from a diffusion model using the Euler-Maruyama method,
with 
- `model` the diffusion model,
- `init_x` as the initial condition,
- `time_steps` the vector of times at which a solution is computed,
   which should advance in ascending order for the forward SDE
   and descending order for the reverse SDE,
- `Δt` the absolute value of the timestep,
- `c` the contextual fields,
- `forward` a boolean indicating if the forward or reverse SDE is used.
# References
https://arxiv.org/abs/1505.04597
"""
function Euler_Maruyama_sampler(model::CliMAgen.AbstractDiffusionModel,
                                init_x::A,
                                time_steps,
                                Δt;
                                c=nothing,
                                forward = false
                                )::A where {A}
    x = mean_x = init_x

    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)
        if forward
            x = x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
        else
        score = CliMAgen.score(model, x, batch_time_step; c=c)
        mean_x = x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt
        x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
        end
    end
    return x
end

"""
    Euler_Maruyama_ld_sampler(model::CliMAgen.AbstractDiffusionModel,
                           init_x::A,
                           time_steps,
                           Δt;
                           c=nothing,
                           forward = false
                           )::A where {A}

Generate a sample from a diffusion model using the Euler-Maruyama method,
using the large-deviation drift, 
with 
- `model` the diffusion model,
- `init_x` as the initial condition,
- `time_steps` the vector of times at which a solution is computed,
   which should advance in ascending order for the forward SDE
   and descending order for the reverse SDE,
- `Δt` the absolute value of the timestep,
- `c` the contextual fields,
- `forward` a boolean indicating if the forward or reverse SDE is used.
# References
https://arxiv.org/abs/1505.04597
"""
function Euler_Maruyama_ld_sampler(model::CliMAgen.AbstractDiffusionModel,
                                init_x::A,
                                time_steps,
                                Δt;
                                bias=nothing,
                                c=nothing,
                                forward = false
                                )::A where {A}
    x = mean_x = init_x

    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)
        if forward
            x = x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
        else
            if bias isa Nothing
                score = CliMAgen.score(model, x, batch_time_step; c=c)
            else
                score = CliMAgen.score(model, x, batch_time_step; c=c) .+ bias
            end
        mean_x = x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt
        x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
        end
    end
    return x
end

"""
    predictor_corrector_sampler(model::CliMAgen.AbstractDiffusionModel,
                                init_x::A,
                                time_steps,
                                Δt;
                                snr=0.16f0,
                                c=nothing
                                )::A where{A}
# References
https://yang-song.github.io/blog/2021/score/#how-to-solve-the-reverse-sde
"""
function predictor_corrector_sampler(model::CliMAgen.AbstractDiffusionModel,
                                     init_x::A,
                                     time_steps,
                                     Δt;
                                     snr=0.16f0,
                                     c=nothing
                                     )::A where{A}
    x = mean_x = init_x

    @showprogress "Predictor Corrector Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step

        # Corrector step (Langevin MCMC)
        grad = CliMAgen.score(model, x, batch_time_step; c=c)

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
        grad = CliMAgen.score(model, x, batch_time_step; c= c)

        mean_x = x .+ CliMAgen.expand_dims((g .^ 2), 3) .* grad .* Δt
        x = mean_x + sqrt.(CliMAgen.expand_dims((g .^ 2), 3) .* Δt) .* randn!(similar(x))
    end
    return x
end

"""
    setup_sampler(model::CliMAgen.AbstractDiffusionModel,
                  device,
                  tilesize,
                  noised_channels;
                  num_images = 5,
                  num_steps=500,
                  ϵ=1.0f-3)

Helper function that generates the required input 
for generating samples from a diffusion model using the
the reverse SDE.

Constructs and returns the `time_steps` array,
timestep `Δt`, and the the initial condition
for the diffusion model `model` at t=1. The returned
initial condition is either on the GPU or CPU, according
to the passed function `device  = Flux.gpu` or `device = Flux.cpu`.
"""
function setup_sampler(model::CliMAgen.AbstractDiffusionModel,
                       device,
                       tilesize,
                       noised_channels;
                       num_images = 5,
                       num_steps=500,
                       ϵ=1.0f-3)

    t = ones(Float32, num_images) |> device
    init_z = randn(Float32, (tilesize, tilesize, noised_channels, num_images)) |> device
    _, σ_T = CliMAgen.marginal_prob(model, zero(init_z), t)
    init_x = (σ_T .* init_z)
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    
    return time_steps, Δt, init_x
end
