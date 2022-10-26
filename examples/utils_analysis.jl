using Flux
using StatsBase
using Statistics
using CliMAgen
using CUDA
using Images
using Random
using FFTW
using DifferentialEquations
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
Helper to make a conglomerate image from an batch of images.
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
    elseif inchannels == 2
        x = Gray.(permutedims(vcat(reshape.(Flux.chunk(x[:, :, 1, :] |> cpu, ncolumns), npixels, :)...), (2, 1)))
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
"Helper to set up a DiffEq SDEProblem"
function setup_SDEProblem(model, init_x, nsteps; ϵ=1.0f-5, reverse = false)
    if reverse
        time_steps = LinRange(1.0f0, ϵ, nsteps)
        f,g = CliMAgen.reverse_sde(model)
        Δt = time_steps[1] - time_steps[2]
    else
        time_steps = LinRange(0.0f0, 1.0f0, nsteps)
        f,g = CliMAgen.forward_sde(model)
        Δt = time_steps[2] - time_steps[1]
    end
    tspan = (time_steps[begin], time_steps[end])
    sde_problem = DifferentialEquations.SDEProblem(f, g, init_x, tspan)
    return sde_problem, Δt
end

"Helper to make a noising/denoising gif, using DifferentialEquations"
function model_gif(model, init_x, nsteps, savepath, plotname ; ϵ=1.0f-5, reverse = false, fps = 50, solver = DifferentialEquations.EM(), time_stride = 2)
    sde, Δt = setup_SDEProblem(model, init_x,nsteps; ϵ=ϵ, reverse = reverse)
    solution = DifferentialEquations.solve(sde, solver, dt=Δt)
    animation_images = convert_to_animation(solution.u, time_stride)
    gif(animation_images, joinpath(savepath, plotname); fps = fps)
end

"Helper to make a noising/denoising bridge plot, using DifferentialEquations

The forward and reverse models may have been trained on different data sets.
In the final plot, we've shifted everything to be on the same grayscale [0,1].
"
function bridge_plot(forward_model, reverse_model, init_x, nsteps, savepath, plotname ; ϵ=1.0f-5, solver = DifferentialEquations.EM(), nimages = 4)
    forward_sde, Δt = setup_SDEProblem(forward_model, init_x,nsteps; ϵ=ϵ)
    save_step = (1-ϵ^(1/4))/nimages
    saveat = (ϵ^(1/4):save_step:1).^4
    forward_solution = DifferentialEquations.solve(forward_sde, solver, dt=Δt, saveat = saveat)
    reverse_sde, Δt = setup_SDEProblem(reverse_model, randn!(similar(init_x)), nsteps; ϵ=ϵ, reverse = true)
    reverse_solution = DifferentialEquations.solve(reverse_sde, solver, dt=Δt, saveat = reverse(saveat))
    images = cat(forward_solution.u..., reverse_solution.u[2:end]..., dims = (4));
    maxes = maximum(images[:,:,[1],:], dims = (1,2))
    mins = minimum(images[:,:,[1],:], dims = (1,2))
    images = @. (images[:,:,[1],:] - mins) / (maxes - mins)

    img_plot(images, savepath, plotname; ncolumns = size(images)[end])
end
