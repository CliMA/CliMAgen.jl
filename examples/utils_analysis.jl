using Flux
using StatsBase
using Statistics
using CliMAgen
using CUDA
using Random
using FFTW

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
Helper function to make an image plot.
"""
function img_plot(samples, save_path, plotname, tilesize, nchannels)
    # clip samples to [0, 1] range
    @. samples = max(0, samples)
    @. samples = min(1, samples)

    samples = convert_to_image(samples |> cpu, nchannels, tilesize)
    Images.save(joinpath(save_path, plotname), samples)

end

"""
Helper function to make analyze the means of the samples.
"""
function spatial_mean_plot(data, gen, savepath, plotname; FT=Float32)
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
    
end

"""
Helper function to make a Q-Q plot.
"""
function qq_plot(data, gen, savepath, plotname; FT=Float32)
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
end

"""
Helper function to make a spectrum plot.
"""
function spectrum_plot(data, gen, savepath, plotname; FT=Float32)
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
end

"""
Helper to make an image from an array.
"""
function convert_to_image(x::AbstractArray{T,N}, inchannels, datasize; n_horizontal=10) where {T,N}
    ysize = min(size(x)[end], n_horizontal)
    num_in_plot = size(x)[end] < n_horizontal ? size(x)[end] : div(size(x)[end], n_horizontal)*n_horizontal
    x = x[:,:,:,1:num_in_plot]
    if inchannels == 1
        x = Gray.(permutedims(vcat(reshape.(Flux.chunk(x |> cpu, ysize), datasize, :)...), (2, 1)))
        return x
    elseif inchannels == 2
        x = Gray.(permutedims(vcat(reshape.(Flux.chunk(x[:, :, 1, :] |> cpu, ysize), datasize, :)...), (2, 1)))
        return x
    elseif inchannels == 3
        tmp = reshape.(Flux.chunk(permutedims(x, (3, 2, 1, 4)) |> cpu, ysize), 3, datasize, :)
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
function convert_to_animation(x, hpdata)
    frames = size(x)[end]
    batches = size(x)[end-1]
    animation = @animate for i = 1:frames+frames÷4
        if i <= frames
            heatmap(
                convert_to_image(x[:, :, :, :, i], hpdata.inchannels, hpdata.size),
                title="Iteration: $i out of $frames"
            )
        else
            heatmap(
                convert_to_image(x[:, :, :, :, end], hpdata.inchannels, hpdata.size),
                title="Iteration: $frames out of $frames"
            )
        end
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
