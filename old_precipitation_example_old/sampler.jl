import CliMAgen: setup_sampler

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
                       tilesize::Tuple,
                       noised_channels;
                       num_images = 5,
                       num_steps=500,
                       ϵ=1.0f-3,
                       nspatial=2,
                       ntime=nothing,
                       FT=Float32,
                       )

    t = ones(FT, num_images) |> device
    if nspatial == 2
        init_z = randn(FT, (tilesize[1], tilesize[2], noised_channels, num_images)) |> device
        @assert ntime isa Nothing
    elseif nspatial == 3
        init_z = randn(FT, (tilesize[1], tilesize[2], tilesize[3], noised_channels, num_images)) |> device
    else
        error("$nspatial must be 2 or 3.")
    end
    _, σ_T = CliMAgen.marginal_prob(model, zero(init_z), t)
    init_x = (σ_T .* init_z)
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

function setup_sampler(model::CliMAgen.AbstractDiffusionModel,
                       device,
                       tilesize::AbstractArray,
                       noised_channels;
                       num_images = 5,
                       num_steps=500,
                       ϵ=1.0f-3,
                       nspatial=2,
                       ntime=nothing,
                       FT=Float32,
                       )

        return_val = setup_sampler(model::CliMAgen.AbstractDiffusionModel,
                       device,
                       Tuple(tilesize),
                       noised_channels;
                       num_images = num_images,
                       num_steps=num_steps,
                       ϵ=ϵ,
                       nspatial=nspatial,
                       ntime=ntime,
                       FT=FT,
                       )
    return return_val
end