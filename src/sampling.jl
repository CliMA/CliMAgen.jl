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
function Euler_Maruyama_sampler(model::CliMAgen.VarianceExplodingSDE,
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

function Euler_Maruyama_timeseries_sampler(model::CliMAgen.VarianceExplodingSDE,
                                           init_x::A,
                                           time_steps,
                                           Δt;
                                           c::A,
                                           forward = false
                                           )::A where {A}
    x = similar(init_x)
    nbatch = size(init_x)[end]
    
    # Stepping in diffusion time
    @assert size(c)[end]==1
    @showprogress "Euler-Maruyama Sampling" for b in 1:nbatch
        x[:,:,:,[b]] .= init_x[:,:,:,[b]]
        for time_step in time_steps
             batch_time_step = [time_step]
            g = CliMAgen.diffusion(model, batch_time_step)
            if forward
                x[:,:,:,[b]] .= x[:,:,:,[b]] .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x[:,:,:,[b]]))
            else
                if b == 1
                    score = CliMAgen.score(model, x[:,:,:,[b]], batch_time_step; c=c)
                else
                    score = CliMAgen.score(model, x[:,:,:,[b]], batch_time_step; c=x[:,:,:,[b-1]])
                end
                x[:,:,:,[b]] .= x[:,:,:,[b]] .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x[:,:,:,[b]]))
            end
        end
    end
    return x
end


function Euler_Maruyama_sampler(model::CliMAgen.VarianceExplodingSDE{ConditionalDiffusiveEstimator},
                                init_x::A,
                                time_steps,
                                Δt;
                                c::A,
                                forward = false
                                )::A where {A}
    x = mean_x = init_x
    t = time_steps[1]
    x_channels = size(init_x)[end-1]
    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)
        
        if forward
            x = x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
        else
            # Noise the context to the current time t
            z = randn!(similar(c))
            μ_t, σ_t = marginal_prob(model, c, t)
            c_t = @. μ_t + σ_t * z
            z = cat(x, c_t, dims = 3)
            
            score = CliMAgen.score(model, z, batch_time_step) # = (s_x, s_c)
            
            mean_x = x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score[:,:,1:x_channels,:] .* Δt
            x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
            t = t - Δt
        end
    end
    return x
end

function Euler_Maruyama_timeseries_sampler(model::CliMAgen.VarianceExplodingSDE{ConditionalDiffusiveEstimator},
                                           init_x::A,
                                           time_steps,
                                           Δt;
                                           c::A,
                                           forward = false
                                           )::A where {A}
    x = mean_x = init_x
    t = time_steps[1]
    x_channels = size(init_x)[end-1]
    nbatch = size(init_x)[end]
    # Stepping in diffusion time
    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = CliMAgen.diffusion(model, batch_time_step)

        if forward
            x = x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
        else
            for b in 1:nbatch
            if b == 1
                # Noise the context to the current time t
                z = randn!(similar(c[:, :, :, [1]]))
                μ_t, σ_t = marginal_prob(model, c[:, :, :, [1]], t)
                c_t = @. μ_t + σ_t * z
            else
                c_t = x[:,:,:,[b-1]]
            end
            y = cat(x[:, :, :, [b]], c_t, dims = 3) # (x,c)

            score = CliMAgen.score(model, y, batch_time_step[[b]]) # = (s_x, s_c)

            mean_x[:, :, :, [b]] = x[:, :, :, [b]] .+ CliMAgen.expand_dims(g[[b]], 3) .^ 2 .* score[:,:,1:x_channels,:] .* Δt
            x[:, :, :, [b]] = mean_x[:, :, :, [b]] .+ sqrt(Δt) .* CliMAgen.expand_dims(g[[b]], 3) .* randn!(similar(x[:, :, :, [b]]))
            t = t - Δt
            end
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
