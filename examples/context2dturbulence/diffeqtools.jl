# Note that by default this is not in the Project.toml
# for experiments because DiffEq takes so long to precompile
# and is only needed rarely.
using DifferentialEquations

"""
    setup_SDEProblem(model::CliMAgen.AbstractDiffusionModel, init_x, nsteps::Int; ϵ=1.0f-5, reverse::Bool=false, t_end=1.0f0)

Creates and returns a DifferentialEquations.SDEProblem object corresponding to
the forward or reverse SDE specific by the `model`, with initial condition `init_x`,
an integration timespan of `[ϵ, t_end]`, and with `nsteps` timesteps to be taken 
during the integration. 

If `reverse` is true, the integration uses the reverse SDE of the
model, and the integration proceeds from `t_end` to `ϵ`, whereas if 
`reverse` is false, the integration uses the forward SDE of the model,
and the integration proceeds from `ϵ` to `t_end`.

The timestep `Δt `corresponding to this setup is also returned. This is a positive
quantity by defintion.
"""
function setup_SDEProblem(model::CliMAgen.AbstractDiffusionModel, init_x, nsteps::Int; c=nothing,ϵ=1.0f-5, reverse::Bool=false, t_end=1.0f0)
    if reverse
        time_steps = LinRange(t_end, ϵ, nsteps)
        f,g = CliMAgen.reverse_sde(model)
        Δt = time_steps[1] - time_steps[2]
    else
        time_steps = LinRange(ϵ, t_end, nsteps)
        f,g = CliMAgen.forward_sde(model)
        Δt = time_steps[2] - time_steps[1]
    end
    tspan = (time_steps[begin], time_steps[end])
    sde_problem = DifferentialEquations.SDEProblem(f, g, init_x, tspan, c)
    return sde_problem, Δt
end

"""
    setup_ODEProblem(model::CliMAgen.AbstractDiffusionModel, init_x, nsteps::Int; ϵ=1.0f-5, reverse::Bool=false, t_end=1.0f0)

Creates and returns a DifferentialEquations.ODEProblem object corresponding to
the probablity flow ODE specific by the `model`, with initial condition `init_x`,
an integration timespan of `[ϵ, t_end]`, and with `nsteps` timesteps to be taken 
during the integration. 

If `reverse` is true, the integration proceeds from `t_end` to `ϵ`, whereas if 
`reverse` is false, the integration proceeds from `ϵ` to `t_end`. The same
tendency is used in either case.

The timestep `Δt `corresponding to this setup is also returned. This is a positive
quantity by defintion.
"""
function setup_ODEProblem(model::CliMAgen.AbstractDiffusionModel, init_x, nsteps::Int; c=nothing,ϵ=1.0f-5, reverse::Bool=false, t_end=1.0f0)
    if reverse
        time_steps = LinRange(t_end, ϵ, nsteps)
        f= CliMAgen.probability_flow_ode(model)
        Δt = time_steps[1] - time_steps[2]
    else
        time_steps = LinRange(ϵ, t_end, nsteps)
        f= CliMAgen.probability_flow_ode(model)
        Δt = time_steps[2] - time_steps[1]
    end
    tspan = (time_steps[begin], time_steps[end])
    ode_problem = DifferentialEquations.ODEProblem(f, init_x, tspan, c)
    return ode_problem, Δt
end

"""
    model_gif(model, init_x, nsteps, savepath, plotname;
              ϵ=1.0f-5, reverse=false, fps=50, sde=true, solver=DifferentialEquations.EM(), time_stride=2)

Creates a gif showing the noising (`reverse=false`) or denoising (`reverse=true`) process given a `model`,
 an initial condition at `t=ϵ` (noising) or `t = 1.0` (denoising) of `init_x`. 
During the integration, `nsteps` are taken, and the resulting animation shows the 
results with a `time_stride` and frames per second of `fps`. 

For example, if `n_steps= 300`, and `time_stride = 3`, 100 images will be shown during the animation.
If `fps = 10`, the resulting gif will take 10 seconds to play. 

The integration can be carried out using either the SDE or the ODE of the model, uses DifferentialEquations,
and uses the DifferentialEquations solver passed in via the `solver` kwarg. If you wish to use a different solver,
you willy likely need to import it directly from DifferentialEquations.
"""
function model_gif(model::CliMAgen.AbstractDiffusionModel, init_x, nsteps::Int, savepath::String, plotname::String;
                   c=nothing, ϵ=1.0f-5, reverse::Bool=false, fps=50, sde::Bool=true, solver=DifferentialEquations.EM(), time_stride::Int=2)
    if sde
        de, Δt = setup_SDEProblem(model, init_x, nsteps; c=c,ϵ=ϵ, reverse=reverse)
    else
        de, Δt = setup_ODEProblem(model, init_x, nsteps; c=c, ϵ=ϵ, reverse=reverse)
    end
    solution = DifferentialEquations.solve(de, solver, dt=Δt)
    animation_images = convert_to_animation(solution.u, time_stride)
    gif(animation_images, joinpath(savepath, plotname); fps = fps)
end

"""
    t_cutoff(power::FT, k::FT, N::FT, σ_max::FT, σ_min::FT) where {FT}

Computes and returns the time `t` at which the power of 
the radially averaged Fourier spectrum of white noise of size NxN, 
with variance σ_min^2(σ_max/σ_min)^(2t), at wavenumber `k`,
is equal to `power`.
"""
function t_cutoff(power::FT, k::FT, N::FT, σ_max::FT, σ_min::FT) where {FT}
    return 1/2*log(power*N^2/σ_min^2)/log(σ_max/σ_min)
end

"""
    diffusion_simulation(model::CliMAgen.AbstractDiffusionModel,
                         init_x,
                         nsteps::Int;
                         c=nothing,
                         reverse::Bool=false,
                         FT=Float32,
                         ϵ=1.0f-5,
                         sde::Bool=false,
                         solver=DifferentialEquations.RK4(),
                         t_end=1.0f0,
                         nsave::Int=4)

Carries out a numerical simulation of the diffusion process specified
by `model`, for the times `t ∈ [ϵ, t_end], given initial condition `init_x` 
at `t=ϵ`. Setting `reverse` to true implies the simulation proceeds from
t=t_end to t=ϵ. 

The user has the choice of whether or not to use the
stochastic differential equation or the probability flow ODE of the model,
via the `sde` kwarg, 
and consequently also has the option of choosing the `DifferentialEquations`
solver as well.

Adaptive timestepping is not supported, because the type of the floats used
is not maintained by DifferentialEquations in this case.
Therefore, the user also specifes the timestep implicitly by choosing `nsteps`.

Lastly, the user specifies how many output images to save (`nsave`) and return.

Returns a DifferentialEquations solution object, with fields `t` and `u`.
"""
function diffusion_simulation(model::CliMAgen.AbstractDiffusionModel,
                              init_x,
                              nsteps::Int;
                              c=nothing,
                              reverse::Bool=false,
                              FT=Float32,
                              ϵ=1.0f-5,
                              sde::Bool=false,
                              solver=DifferentialEquations.RK4(),
                              t_end=1.0f0,
                              nsave::Int=4)
    # Visually, stepping linearly in t^(1/2 seemed to generate a good
    # picture of the noising process. 
    start = sqrt(ϵ)
    stop = sqrt(t_end)
    saveat = FT.(range(start, stop, length = nsave)).^2
    if reverse
        saveat = Base.reverse(saveat)
    end

    # Pad end time slightly to make sure we integrate and save the solution at t_end
    if sde
        de, Δt = setup_SDEProblem(model, init_x, nsteps; c=c, ϵ=ϵ, reverse = reverse, t_end = t_end*FT(1.01))
    else
        de, Δt = setup_ODEProblem(model, init_x, nsteps; c=c, ϵ=ϵ, reverse = reverse, t_end = t_end*FT(1.01))
    end
    solution = DifferentialEquations.solve(de, solver, dt=Δt, saveat = saveat, adaptive = false)
    return solution
end


"""
    diffusion_bridge_simulation(forward_model::CliMAgen.VarianceExplodingSDE,
                                reverse_model::CliMAgen.VarianceExplodingSDE,
                                init_x,
                                nsteps::Int,
                                ;
                                forward_c=nothing,
                                reverse_c=nothing,
                                FT=Float32,
                                ϵ=1.0f-5,
                                forward_sde::Bool=false,
                                reverse_sde::Bool=false,
                                forward_solver=DifferentialEquations.RK4(),
                                reverse_solver=DifferentialEquations.RK4(),
                                forward_t_end=1.0f0,
                                reverse_t_end=1.0f0,
                                nsave::Int=4)

Carries out a diffusion bridge simulation and returns the trajectory.

In the first leg, `forward_model` is used to integrate
from t=ϵ to t=forward_t_end, using the `forward_solver` for
timestepping, according to the SDE or ODE depending on the
choice for `forward_sde`. In the reverse leg, the corresponding
is true.

Before beginning the reverse leg, the last output of the
forward leg is adapted, as indicated by the chosen noising
schedule of each model (which may differ).

Only fixed timestep methods are used; `nsteps` implicitly
determines the timestep, and `nsave` determines how many
timesteps are saved and returned per leg of the diffusion 
bridge.
"""
function diffusion_bridge_simulation(forward_model::CliMAgen.VarianceExplodingSDE,
                                     reverse_model::CliMAgen.VarianceExplodingSDE,
                                     init_x,
                                     nsteps::Int,
                                     ;
                                     forward_c=nothing,
                                     reverse_c=nothing,
                                     FT=Float32,
                                     ϵ=1.0f-5,
                                     forward_sde::Bool=false,
                                     reverse_sde::Bool=false,
                                     forward_solver=DifferentialEquations.RK4(),
                                     reverse_solver=DifferentialEquations.RK4(),
                                     forward_t_end=1.0f0,
                                     reverse_t_end=1.0f0,
                                     nsave::Int=4)
    
    forward_solution = diffusion_simulation(forward_model, init_x, nsteps;
                                            c=forward_c,
                                            reverse=false,
                                            FT=FT,
                                            ϵ=ϵ,
                                            sde=forward_sde,
                                            solver=forward_solver,
                                            t_end=forward_t_end,
                                            nsave=nsave)
    init_x_reverse = forward_solution.u[end]
    adapt_x!(init_x_reverse, forward_model, reverse_model, forward_t_end, reverse_t_end)

    reverse_solution = diffusion_simulation(reverse_model, init_x_reverse,  nsteps;
                                            c=reverse_c,
                                            reverse=true,
                                            FT=FT,
                                            ϵ=ϵ,
                                            sde=reverse_sde,
                                            solver=reverse_solver,
                                            t_end=reverse_t_end,
                                            nsave=nsave)
    return forward_solution, reverse_solution
end
