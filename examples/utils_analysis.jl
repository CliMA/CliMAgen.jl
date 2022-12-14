using Flux
using StatsBase
using Statistics
using CliMAgen
using CUDA
using Images
using Random
using FFTW
using DifferentialEquations
using DelimitedFiles

"""
    loss_plot(savepath::String, plotname::String; xlog::Bool=false, ylog::Bool=true)

Creates and saves a plot of the training and test loss values, for both the spatial
and mean loss terms. Whether or not the axes are linear or logarithmic is controlled
by the `xlog` and `ylog` boolean keyword arguments.

The saved plot can be found at `joinpath(savepath,plotname)`.
"""
function loss_plot(savepath::String, plotname::String; xlog::Bool=false, ylog::Bool=true)
    path = joinpath(savepath,plotname)
    filename = joinpath(savepath, "losses.txt")
    data = DelimitedFiles.readdlm(filename, ',', skipstart = 1)

     plt1 = plot(left_margin = 20Plots.mm, ylabel = "Log10(Mean Loss)")
    plt2 = plot(bottom_margin = 10Plots.mm, left_margin = 20Plots.mm,xlabel = "Epoch", ylabel = "Log10(Spatial Loss)")
    plot!(plt1, data[:,1], data[:,2], label = "Train", linecolor = :black)
    plot!(plt1, data[:,1], data[:,4], label = "Test", linecolor = :red)
    plot!(plt2, data[:,1], data[:,3], label = "", linecolor = :black)
    plot!(plt2, data[:,1], data[:,5], label = "", linecolor = :red)
    if xlog
        plot!(plt1, xaxis=:log)
        plot!(plt2, xaxis=:log)
    end
    if ylog
        plot!(plt1, yaxis=:log)
        plot!(plt2, yaxis=:log)
    end
    plot(plt1, plt2, layout =(2,1))
    savefig(path)
end


"""
Helper function to make an image plot.
"""
function img_plot(samples, savepath, plotname; ncolumns = 10,FT=Float32, logger=nothing)
    # clip samples to [0, 1] range
    @. samples = max(0, samples)
    @. samples = min(1, samples)

    samples = image_grid(samples; ncolumns = ncolumns)
    Images.save(joinpath(savepath, plotname), samples)

    if !(logger isa Nothing)
        CliMAgen.log_artifact(logger, joinpath(savepath, plotname); name=plotname, type="PNG-file")
    end
end

"""
Helper function to make analyze the means of the samples.
"""
function spatial_mean_plot(data, gen, savepath, plotname; FT=Float32, logger=nothing)
    inchannels = size(data)[end-1]

    gen = gen |> Flux.cpu
    gen_results = mapslices(Statistics.mean, gen, dims=[1, 2])
    gen_results = gen_results[1,1,:,:]

    data_results = mapslices(Statistics.mean, data, dims=[1, 2])
    data_results = data_results[1,1,:,:]
    plot_array = []
    for channel in 1:inchannels
        plt = plot(xlabel = "Spatial Mean", ylabel = "Probability density", title = string("Ch:",string(channel)))
        plot!(plt, data_results[channel,:], seriestype=:stephist, label = "data", norm = true, color = :red)
        plot!(plt, gen_results[channel,:],  seriestype=:stephist, label ="generated", norm = true, color = :black)
        push!(plot_array, plt)
    end
    
    plot(plot_array..., layout=(1, inchannels))
    Plots.savefig(joinpath(savepath, plotname))

    if !(logger isa Nothing)
        CliMAgen.log_artifact(logger, joinpath(savepath, plotname); name=plotname, type="PNG-file")
    end
end

"""
Helper function to make a Q-Q plot.
"""
function qq_plot(data, gen, savepath, plotname; FT=Float32, logger=nothing)
    statistics = (Statistics.var, x -> StatsBase.cumulant(x[:], 3), x -> StatsBase.cumulant(x[:], 4))
    statistic_names = ["σ²", "κ₃", "κ₄"]
    inchannels = size(data)[end-1]

    gen = gen |> Flux.cpu
    gen_results = mapslices.(statistics, Ref(gen), dims=[1, 2])
    gen_results = cat(gen_results..., dims=ndims(gen) - 2)
    sort!(gen_results, dims=ndims(gen_results)) # CDF of the generated data for each channel and each statistics


    data_results = mapslices.(statistics, Ref(data), dims=[1, 2])
    data_results = cat(data_results..., dims=ndims(data) - 2)
    sort!(data_results, dims=ndims(data_results)) # CDF of the  data for each channel and each statistics
    plot_array = []
    for channel in 1:inchannels
        for stat in 1:length(statistics)
            data_cdf = data_results[1, stat, channel, :]
            gen_cdf = gen_results[1, stat, channel, :]
            plt = plot(gen_cdf, data_cdf, color=:red, label="")
            plot!(plt, data_cdf, data_cdf, color=:black, linestyle=:dot, label="")
            plot!(plt,
                xlabel="Gen",
                ylabel="Data",
                title=string("Ch:", string(channel), ", ", statistic_names[stat]),
                tickfontsize=4)
            push!(plot_array, plt)
        end
    end

    plot(plot_array..., layout=(inchannels, length(statistics)), aspect_ratio=:equal)
    Plots.savefig(joinpath(savepath, plotname))

    if !(logger isa Nothing)
        CliMAgen.log_artifact(logger, joinpath(savepath, plotname); name=plotname, type="PNG-file")
    end
end

"""
    batch_spectra(data, L::Int)

Computes and returns the mean radially average power 
spectrum for the data, where the mean is taken
over the batch dimension, not the channel dimension,
as well as the spatial frequencies `k`.

This has issues with memory for large images.
Revisit!
"""
function batch_spectra(data, L::Int)
    statistics = x -> hcat(power_spectrum2d(x, L)...)
    data = data |> Flux.cpu
    results = mapslices(statistics, data, dims=[1, 2])
    k = results[:, 2, 1, 1]
    results = results[:, 1, :, :]
    spectrum = mean(results, dims=3)
    return spectrum, k
end


"""
Helper function to make a spectrum plot.
"""
function spectrum_plot(data, gen, savepath, plotname; FT=Float32, logger=nothing) 
    L = FT(1) # Eventually a physical size
    statistics = x -> hcat(power_spectrum2d(x, L)...)
    inchannels = size(data)[end-1]

    data_results = mapslices(statistics, data, dims=[1, 2])
    k = data_results[:, 2, 1, 1]
    data_results = data_results[:, 1, :, :]

    gen = gen |> Flux.cpu
    gen_results = mapslices(statistics, gen, dims=[1, 2])
    gen_results = gen_results[:, 1, :, :]

    plot_array = []
    for channel in 1:inchannels
        data_spectrum = mean(data_results[:, channel, :], dims=2)
        lower_data_spectrum = mapslices(x -> percentile(x[:], 10), data_results[:, channel, :], dims=2)
        upper_data_spectrum = mapslices(x -> percentile(x[:], 90), data_results[:, channel, :], dims=2)
        data_confidence = (data_spectrum .- lower_data_spectrum, upper_data_spectrum .- data_spectrum)
        gen_spectrum = mean(gen_results[:, channel, :], dims=2)
        lower_gen_spectrum = mapslices(x -> percentile(x[:], 10), gen_results[:, channel, :], dims=2)
        upper_gen_spectrum = mapslices(x -> percentile(x[:], 90), gen_results[:, channel, :], dims=2)
        gen_confidence = (gen_spectrum .- lower_gen_spectrum, upper_gen_spectrum .- gen_spectrum)
        plt = plot(k, data_spectrum, ribbon = data_confidence, color=:red, label="", yaxis=:log, xaxis=:log)
        plot!(plt, k, gen_spectrum, ribbon = gen_confidence, color=:blue, label="")
        plot!(plt,
            xlabel="Log(k)",
            ylabel="Log(Power)",
            title=string("Ch:", string(channel)),
            tickfontsize=4)
        push!(plot_array, plt)
    end

    plot(plot_array..., layout=(inchannels, 1))
    Plots.savefig(joinpath(savepath, plotname))

    if !(logger isa Nothing)
        CliMAgen.log_artifact(logger, joinpath(savepath, plotname); name=plotname, type="PNG-file")
    end
end

"""
    image_grid(x::AbstractArray{T,N}; ncolumns=10)

Rearranges  a 1-d array of images (where an image is taken to be m x m x inchannels)
 into a two-dimensional array of images, with ncolumns in the newly created second dimension. 
Converts the images to grayscale (if inchannels is 1) or RGB (if inchannels is 3).

inchannels not equal to 1 or 3 is not supported.
"""
function image_grid(x::AbstractArray{T,N}; ncolumns=10) where {T,N}
    # Number of images per row of the grid
    batchsize = size(x)[end]
    ncolumns = min(batchsize, ncolumns)
    # We want either an even number of images per row
    nimages = div(batchsize, ncolumns)*ncolumns
    x = x[:,:,:,1:nimages]
    
    # Number of pixels per spatial direction of a single image
    npixels = size(x)[1]

    inchannels = size(x)[end-1]
    if inchannels == 1
        x = Gray.(permutedims(vcat(reshape.(Flux.chunk(x |> cpu, ncolumns), npixels, :)...), (2, 1)))
        return x
    elseif inchannels == 3
        tmp = reshape.(Flux.chunk(permutedims(x, (3, 2, 1, 4)) |> cpu, ncolumns), 3, npixels, :)
        rgb = colorview.(Ref(RGB), tmp)
        return vcat(rgb...)
    else
        error("Number of inchannels not supported")
    end
    return x
end

"""
Helper to make an animation from a batch of images.
"""
function convert_to_animation(x, time_stride)
    init_frames = length(x)
    x = x[1:time_stride:init_frames]
    frames = length(x)
    animation = @animate for i = 1:frames
            heatmap(
                image_grid(x[i]),
                xaxis = false, yaxis = false, xticks = false, yticks = false,
            )
    end
    return animation
end

"""
    power_spectrum2d(img, L, dim)

Adapted from https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Common/Spectra/power_spectrum_les.jl
for two spatial dimensions.

Inputs
need to be equi-spaced and the domain is assumed to be the same size and
have the same number of points in all directions.
# Arguments
- img: a 2 dimension matrix of size (N, N).
- L: physical size domain
"""
function power_spectrum2d(img, L)
    @assert size(img)[1] == size(img)[2]
    dim = size(img)[1]
    img_fft = abs.(fft(img .- mean(img)))
    m = Array(img_fft / size(img_fft, 1)^2)
    if mod(dim, 2) == 0
        rx = range(0, stop=dim - 1, step=1) .- dim / 2 .+ 1
        ry = range(0, stop=dim - 1, step=1) .- dim / 2 .+ 1
        R_x = circshift(rx', (1, dim / 2 + 1))
        R_y = circshift(ry', (1, dim / 2 + 1))
        k_nyq = dim / 2
    else
        rx = range(0, stop=dim - 1, step=1) .- (dim - 1) / 2
        ry = range(0, stop=dim - 1, step=1) .- (dim - 1) / 2
        R_x = circshift(rx', (1, (dim + 1) / 2))
        R_y = circshift(ry', (1, (dim + 1) / 2))
        k_nyq = (dim - 1) / 2
    end
    r = zeros(size(rx, 1), size(ry, 1))
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        r[i, j] = sqrt(R_x[i]^2 + R_y[j]^2)
    end
    dx = 2 * pi / L
    k = range(1, stop=k_nyq, step=1) .* dx
    endk = size(k, 1)
    contribution = zeros(endk)
    spectrum = zeros(endk)
    for N in 2:Int64(k_nyq - 1)
        for i in 1:size(rx, 1), j in 1:size(ry, 1)
            if (r[i, j] * dx <= (k'[N+1] + k'[N]) / 2) &&
               (r[i, j] * dx > (k'[N] + k'[N-1]) / 2)
                spectrum[N] =
                    spectrum[N] + m[i, j]^2
                contribution[N] = contribution[N] + 1
            end
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j] * dx <= (k'[2] + k'[1]) / 2)
            spectrum[1] =
                spectrum[1] + m[i, j]^2
            contribution[1] = contribution[1] + 1
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j] * dx <= k'[endk]) &&
           (r[i, j] * dx > (k'[endk] + k'[endk-1]) / 2)
            spectrum[endk] =
                spectrum[endk] + m[i, j]^2
            contribution[endk] = contribution[endk] + 1
        end
    end
    spectrum = spectrum .* 2 .* pi .* k .^ 2 ./ (contribution .* dx .^ 2)

    return spectrum, k
end

"""
    timewise_score_matching_loss(model, x_0, ϵ=1.0f-5)

Compute the loss term for a single realization of data
as a function of time. This is different from the true loss 
term optimized by the network, which takes the expectation of this
quantity over time, training data x(0), and samples from P(x(t)|x(0)).
"""
function timewise_score_matching_loss(model, x_0, ϵ=1.0f-5)
    # sample times
    t = LinRange(0.0f0,1.0f0,size(x_0)[end])

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = CliMAgen.marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = CliMAgen.score(model, x_t, t)

    # Assume that λ(t) = σ(t)² and pull it into L₂-norm
    # Below, z / σ_t = -∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))
    loss = @. (z + σ_t * s_t)^2 # squared deviations from real score
    loss = sum(loss, dims=1:(ndims(x_0)-1)) # L₂-norm

    return t, loss[:]
end
"""
    model_scale(model, x_0, ϵ=1.0f-5)

Compute the scaled score for a single realization of data
as a function of time. 
"""
function model_scale(model, x_0, ϵ=1.0f-5)
    # sample times
    t = LinRange(0.0f0, 1.0f0, size(xtrain)[end])

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = CliMAgen.marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = CliMAgen.score(model, x_t, t)

    # Assume that λ(t) = σ(t)² and pull it into L₂-norm
    # Below, z / σ_t = -∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))
    scale = @. (σ_t * s_t)^2 # squared deviations from real score
    scale = sum(scale, dims=1:(ndims(x_0)-1)) # L₂-norm

    return t, scale[:]
end


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
function setup_SDEProblem(model::CliMAgen.AbstractDiffusionModel, init_x, nsteps::Int; ϵ=1.0f-5, reverse::Bool=false, t_end=1.0f0)
    if reverse
        time_steps = LinRange(t_end, ϵ, nsteps)
        f,g = CliMAgen.reverse_sde(model)
        Δt = time_steps[1] - time_steps[2]
    else
        time_steps = LinRange(0.0f0, t_end, nsteps)
        f,g = CliMAgen.forward_sde(model)
        Δt = time_steps[2] - time_steps[1]
    end
    tspan = (time_steps[begin], time_steps[end])
    sde_problem = DifferentialEquations.SDEProblem(f, g, init_x, tspan)
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
function setup_ODEProblem(model::CliMAgen.AbstractDiffusionModel, init_x, nsteps::Int; ϵ=1.0f-5, reverse::Bool=false, t_end=1.0f0)
    if reverse
        time_steps = LinRange(t_end, ϵ, nsteps)
        f= CliMAgen.probability_flow_ode(model)
        Δt = time_steps[1] - time_steps[2]
    else
        time_steps = LinRange(0.0f0, t_end, nsteps)
        f= CliMAgen.probability_flow_ode(model)
        Δt = time_steps[2] - time_steps[1]
    end
    tspan = (time_steps[begin], time_steps[end])
    ode_problem = DifferentialEquations.ODEProblem(f, init_x, tspan)
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
                   ϵ=1.0f-5, reverse::Bool=false, fps=50, sde::Bool=true, solver=DifferentialEquations.EM(), time_stride::Int=2)
    if sde
        de, Δt = setup_SDEProblem(model, init_x,nsteps; ϵ=ϵ, reverse=reverse)
    else
        de, Δt = setup_ODEProblem(model, init_x,nsteps; ϵ=ϵ, reverse=reverse)
    end
    solution = DifferentialEquations.solve(de, solver, dt=Δt)
    animation_images = convert_to_animation(solution.u, time_stride)
    gif(animation_images, joinpath(savepath, plotname); fps = fps)
end

"""
    t_cutoff(power::FT, k::FT, σ_max::FT, σ_min::FT) where {FT}

Computes and returns the time `t` at which the power of 
the radially averaged Fourier spectrum of white noise, 
with variance σ_min^2(σ_max/σ_min)^(2t), at wavenumber `k`,
is equal to `power`.

If `power` is the power of the radially averaged Fourier spectrum
of a source image, at wavenumber `k`, this time corresponds to the 
approximate time at which the signal to noise at `k` is 1.
This is because the radial power(k) for white noise is: k^2*σ^2/2π,
and the forward diffusion process for a VE difussion mode 
satisfies dx = g(t) dw, with ∫_0^t g(s)^2 ds) = σ_min^2(σ_max/σ_min)^(2t).
"""
function t_cutoff(power::FT, k::FT, σ_max::FT, σ_min::FT) where {FT}
    return 1/2*log(2 * pi * power/k^2/σ_min^2)/log(σ_max/σ_min)
end

"""
    diffusion_simulation(model::CliMAgen.AbstractDiffusionModel,
                         init_x,
                         nsteps::Int;
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

    if sde
        de, Δt = setup_SDEProblem(model, init_x, nsteps; ϵ=ϵ, reverse = reverse, t_end = t_end)
    else
        de, Δt = setup_ODEProblem(model, init_x, nsteps; ϵ=ϵ, reverse = reverse, t_end = t_end)
    end
    solution = DifferentialEquations.solve(de, solver, dt=Δt, saveat = saveat, adaptive = false)
    return solution
end

"""
    adapt_x(x,
            forward_model::CliMAgen.VarianceExplodingSDE,
            reverse_model::CliMAgen.VarianceExplodingSDE,
            forward_t_end::FT,
            reverse_t_end::FT) where{FT}

Adapts the state `x` produced by diffusion to `forward_t_end`
from `t=0`, using the `forward_model` to an equivalent state produced by
`reverse_model`` after integrating to `reverse_t_end` from `t=1`.

Useful for diffusion bridges between datasets generated by 
Variance Exploding SDE models with different values of
σ_max and σ_min.
"""
function adapt_x(x,
                 forward_model::CliMAgen.VarianceExplodingSDE,
                 reverse_model::CliMAgen.VarianceExplodingSDE,
                 forward_t_end::FT,
                 reverse_t_end::FT) where{FT}
    _, forward_σ_end = CliMAgen.marginal_prob(forward_model, x, FT(forward_t_end)) # x only affects the mean, which we dont use
    _, reverse_σ_end = CliMAgen.marginal_prob(reverse_model, x, FT(reverse_t_end)) # x only affects the mean, which we dont use
    init_x = x .* reverse_σ_end ./ forward_σ_end
    return init_x
end

"""
    diffusion_bridge_simulation(forward_model::CliMAgen.VarianceExplodingSDE,
                                reverse_model::CliMAgen.VarianceExplodingSDE,
                                init_x,
                                nsteps::Int,
                                ; FT=Float32,
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
                                     ; FT=Float32,
                                     ϵ=1.0f-5,
                                     forward_sde::Bool=false,
                                     reverse_sde::Bool=false,
                                     forward_solver=DifferentialEquations.RK4(),
                                     reverse_solver=DifferentialEquations.RK4(),
                                     forward_t_end=1.0f0,
                                     reverse_t_end=1.0f0,
                                     nsave::Int=4)
    
    forward_solution = diffusion_simulation(forward_model, init_x, nsteps;
                                            reverse=false,
                                            FT=FT,
                                            ϵ=ϵ,
                                            sde=forward_sde,
                                            solver=forward_solver,
                                            t_end=forward_t_end,
                                            nsave=nsave)

    init_x_reverse = adapt_x(forward_solution.u[end], forward_model, reverse_model, forward_t_end, reverse_t_end)

    reverse_solution = diffusion_simulation(reverse_model, init_x_reverse, nsteps;
                                            reverse=true,
                                            FT=FT,
                                            ϵ=ϵ,
                                            sde=reverse_sde,
                                            solver=reverse_solver,
                                            t_end=reverse_t_end,
                                            nsave=nsave)
    return forward_solution, reverse_solution
end