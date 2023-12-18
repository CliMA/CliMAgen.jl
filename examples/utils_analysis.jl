using Flux
using StatsBase
using Statistics
using CliMAgen
using CUDA
using Images
using Random
using FFTW
using DelimitedFiles
using Plots

"""
    autocorrelation_plot(data, gen, savepath, plotname; FT=Float32, logger=nothing)

Creates and saves a plot of the autocorrelation function for the timeseries passed:
data, gen, both assumed to be of size nsteps x nbatch. The
the plot is saved at joinpath(savepath, plotname).

Uncertainties are estimated using the members of the batch, and the 5th and 95th 
percentiles are plotted along with the mean (across the batch) autocorrelation 
function.
"""
function autocorrelation_plot(data, gen, savepath, plotname; FT=Float32, logger=nothing)
    nsteps = size(gen)[1]
    lags = Array(1:1:nsteps-3)
    ac = mapslices(x -> StatsBase.autocor(x, lags; demean=true), gen, dims=1)
    mean_ac = mean(ac, dims=2)[:]
    upper_ac = mapslices(x -> percentile(x, 95), ac, dims = 2)[:] 
    lower_ac = mapslices(x -> percentile(x, 5), ac, dims = 2)[:] 
    Plots.plot(lags, mean_ac,  ribbon = (mean_ac .- lower_ac, upper_ac .- mean_ac), label = "Generated")

    ac = mapslices(x -> StatsBase.autocor(x, lags; demean=true),data, dims=1)
    mean_ac = mean(ac, dims=2)[:]
    upper_ac = mapslices(x -> percentile(x, 95), ac, dims = 2)[:] 
    lower_ac = mapslices(x -> percentile(x, 5), ac, dims = 2)[:]
    Plots.plot!(lags, mean_ac,  ribbon = (mean_ac .- lower_ac, upper_ac .- mean_ac), label = "Training", ylabel = "Autocorrelation Coeff", xlabel = "Lag (steps)", margin = 10Plots.mm)
    Plots.savefig(joinpath(savepath, plotname))
end

"""
    compute_sigma_max(x)

   Returns σ_max for the dataset `x`, which
is assumed to be of size nx x ny x nc x n_obs.
"""
function compute_sigma_max(x)
    n_obs = size(x)[end]
    max_distance = 0
    for i in 1:n_obs
        for j in i+1:n_obs
            distance = sqrt(sum((x[:,:,:,i] .- x[:,:,:,j]).^2))
            max_distance = max(max_distance, distance)
        end
    end
    return max_distance
end



"""
    loss_plot(savepath::String, plotname::String; xlog::Bool=false, ylog::Bool=true)

Creates and saves a plot of the training and test loss values, for both the spatial
and mean loss terms; creates a saves a plot of the training and test loss values
for the total loss, if using the vanilla loss function. Which option is carried out
depends on the number of columns in the data file: 5 for the split loss function, and 3
for the vanilla loss function.

Whether or not the axes are linear or logarithmic is controlled
by the `xlog` and `ylog` boolean keyword arguments. The saved plot can be found at `joinpath(savepath,plotname)`.
"""
function loss_plot(savepath::String, plotname::String; xlog::Bool=false, ylog::Bool=true)
    path = joinpath(savepath,plotname)
    filename = joinpath(savepath, "losses.txt")
    data = DelimitedFiles.readdlm(filename, ',', skipstart = 1)
    
    if size(data)[2] == 5
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
    elseif size(data)[2] == 3
        plt1 = plot(left_margin = 20Plots.mm, ylabel = "Log10(Loss)")
	plot!(plt1, data[:,1], data[:,2], label = "Train", linecolor = :black)
    	plot!(plt1, data[:,1], data[:,3], label = "Test", linecolor = :red)
    	if xlog
           plot!(plt1, xaxis=:log)
    	end
    	if ylog
           plot!(plt1, yaxis=:log)
        end
	savefig(path)
    else
        @info "Loss CSV file has incorrect number of columns"
    end
end


"""
    heatmap_grid(samples, ch, savepath, plotname; ncolumns = 10,FT=Float32, logger=nothing)

Creates a grid of images with `ncolumns` using the data `samples`. 
Saves the resulting plot at joinpath(savepath,plotname).

"""
function heatmap_grid(samples, ch, savepath, plotname; ncolumns = 5,FT=Float32, logger=nothing)
    batchsize = size(samples)[end]
    ncolumns = min(batchsize, ncolumns)
    # We want either an even number of images per row
    nrows = div(batchsize, ncolumns)
    nimages = nrows*ncolumns
    clims = (minimum(samples), maximum(samples))
    plts = []
    for img in 1:nimages
        push!(plts, Plots.heatmap(samples[:,:,ch,img], aspect_ratio=:equal, clims = clims, border = :box, legend = :none, axis=([], false)))
    end
    Plots.plot(plts..., layout = (nrows, ncolumns), size = (ncolumns*200, nrows*200))
    Plots.savefig(joinpath(savepath, plotname))

    if !(logger isa Nothing)
        CliMAgen.log_artifact(logger, joinpath(savepath, plotname); name=plotname, type="PNG-file")
    end
end



"""
    img_plot(samples, savepath, plotname; ncolumns = 10,FT=Float32, logger=nothing)

Creates a grid of images with `ncolumns` using the data`samples`. 
Saves the resulting plot at joinpath(savepath,plotname).

Note that the samples are clipped to lie within the [0,1] range prior to plotting.
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
    spatial_mean_plot(data, gen, savepath, plotname; FT=Float32, logger=nothing)

Creates and saves histogram plots of the spatial means of `data` and `gen`;
the plot is saved at joinpath(savepath, plotname). Both `data` and `gen`
are assumed to be of size (Nx, Ny, Nchannels, Nbatch).
"""
function spatial_mean_plot(data, gen, savepath, plotname; FT=Float32, nspatial=2, logger=nothing)
    @assert nspatial <= 3
    inchannels = size(data)[end-1]

    if nspatial == 3
        sz = size(data)
        data_rs = reshape(data, (sz[1], sz[2], sz[4], sz[3]*sz[5]))
        gen_rs = reshape(gen, (sz[1], sz[2], sz[4], sz[3]*sz[5]))
    else
        data_rs = data
        gen_rs = gen 
    end

    gen_rs = gen_rs |> Flux.cpu
    gen_results = mapslices(Statistics.mean, gen_rs, dims=[1, 2])
    gen_results = gen_results[1,1,:,:]

    data_results = mapslices(Statistics.mean, data_rs, dims=[1, 2])
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
    qq_plot(data, gen, savepath, plotname; FT=Float32, logger=nothing)

Creates and saves qq plots of the higher order cumulants of `data` and `gen`;
the plot is saved at joinpath(savepath, plotname). Both `data` and `gen`
are assumed to be of size (Nx, Ny, Nchannels, Nbatch).
"""
function qq_plot(data, gen, savepath, plotname; FT=Float32, nspatial=2, logger=nothing)
    @assert nspatial <= 3
    statistics = (Statistics.var, x -> StatsBase.cumulant(x[:], 3), x -> StatsBase.cumulant(x[:], 4))
    statistic_names = ["σ²", "κ₃", "κ₄"]
    inchannels = size(data)[end-1]

    if nspatial == 3
        sz = size(data)
        data_rs = reshape(data, (sz[1], sz[2], sz[4], sz[3]*sz[5]))
        gen_rs = reshape(gen, (sz[1], sz[2], sz[4], sz[3]*sz[5]))
    else
        data_rs = data
        gen_rs = gen 
    end

    gen_rs = gen_rs |> Flux.cpu
    gen_results = mapslices.(statistics, Ref(gen_rs), dims=[1, 2])
    gen_results = cat(gen_results..., dims=ndims(gen_rs) - 2)
    sort!(gen_results, dims=ndims(gen_results)) # CDF of the generated data for each channel and each statistics


    data_results = mapslices.(statistics, Ref(data_rs), dims=[1, 2])
    data_results = cat(data_results..., dims=ndims(data_rs) - 2)
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
    batch_spectra(data)

Computes and returns the mean azimuthally averaged power 
spectrum for the data, where the mean is taken
over the batch dimension,
but not over the channel dimension.
"""
function batch_spectra(data)
    statistics = x -> hcat(power_spectrum2d(x)...)
    data = data |> Flux.cpu
    results = mapslices(statistics, data, dims=[1, 2])
    k = results[:, 2, 1, 1]
    results = results[:, 1, :, :]
    spectrum = mean(results, dims=3)
    return spectrum, k
end


"""
    spectrum_plot(data, gen, savepath, plotname; FT=Float32, logger=nothing)

Creates and saves power spectral density plots by channel for `data` and `gen`;
the plot is saved at joinpath(savepath, plotname). 

Both `data` and `gen` are assumed to be of size (Nx, Ny, Nchannels, Nbatch).
Confidence intervals are computed using the difference batch members as different
samples.
"""
function spectrum_plot(data, gen, savepath, plotname; FT=Float32, nspatial=2, logger=nothing) 
    @assert nspatial <= 3
    inchannels = size(data)[end-1]
    statistics = x -> hcat(power_spectrum2d(x)...)

    if nspatial == 3
        sz = size(data)
        data_rs = reshape(data, (sz[1], sz[2], sz[4], sz[3]*sz[5]))
        gen_rs = reshape(gen, (sz[1], sz[2], sz[4], sz[3]*sz[5]))
    else
        data_rs = data
        gen_rs = gen 
    end    

    data_results = mapslices(statistics, data_rs, dims=[1, 2])
    k = data_results[:, 2, 1, 1]
    data_results = data_results[:, 1, :, :]

    gen_rs = gen_rs |> Flux.cpu
    gen_results = mapslices(statistics, gen_rs, dims=[1, 2])
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
        plt = plot(log2.(k), data_spectrum, ribbon = data_confidence, color=:red, label="", yaxis=:log)
        plot!(plt, log2.(k), gen_spectrum, ribbon = gen_confidence, color=:blue, label="")
        plot!(plt,
            xlabel="Log2(k)",
            ylabel="Log10(Power)",
            title=string("Ch:", string(channel)),
            tickfontsize=4)
        push!(plot_array, plt)
    end

    plot(plot_array..., layout=(inchannels, 1))
    Plots.savefig(joinpath(savepath, plotname))
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
    power_spectrum2d(img)

Adapted from https://github.com/CliMA/ClimateMachine.jl/blob/master/src/Common/Spectra/power_spectrum_les.jl
for two spatial dimensions.

Inputs need to be equi-spaced and the domain is assumed to be the same size and
have the same number of points in all directions.

# Arguments
- img: a 2 dimension matrix of size (N, N).

# Returns
 - spectrum, wavenumber
"""
function power_spectrum2d(img)
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
    k = range(1, stop=k_nyq, step=1)
    endk = size(k, 1)
    contribution = zeros(endk)
    spectrum = zeros(endk)
    for N in 2:Int64(k_nyq - 1)
        for i in 1:size(rx, 1), j in 1:size(ry, 1)
            if (r[i, j] <= (k'[N+1] + k'[N]) / 2) &&
               (r[i, j] > (k'[N] + k'[N-1]) / 2)
                spectrum[N] =
                    spectrum[N] + m[i, j]^2
                contribution[N] = contribution[N] + 1
            end
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j] <= (k'[2] + k'[1]) / 2)
            spectrum[1] =
                spectrum[1] + m[i, j]^2
            contribution[1] = contribution[1] + 1
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j] <= k'[endk]) &&
           (r[i, j] > (k'[endk] + k'[endk-1]) / 2)
            spectrum[endk] =
                spectrum[endk] + m[i, j]^2
            contribution[endk] = contribution[endk] + 1
        end
    end
    spectrum = spectrum ./ contribution

    return spectrum, k
end

"""
    timewise_score_matching_loss(model, x_0, ϵ=1.0f-5)

Compute the total loss term for a single realization of data
as a function of time. This is different from the true loss 
term optimized by the network, which takes the expectation of this
quantity over time, training data x(0), and samples from P(x(t)|x(0)).
"""
function timewise_score_matching_loss(model, x_0; ϵ=1.0f-5, c=nothing)
    # sample times
    t = LinRange(ϵ,1.0f0,size(x_0)[end])

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = CliMAgen.marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = CliMAgen.score(model, x_t, t; c=c)

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
function model_scale(model, x_0; ϵ=1.0f-5, c=nothing)
    # sample times
    t = LinRange(ϵ, 1.0f0, size(xtrain)[end])

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = CliMAgen.marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = CliMAgen.score(model, x_t, t;c=c)

    # Assume that λ(t) = σ(t)² and pull it into L₂-norm
    # Below, z / σ_t = -∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))
    scale = @. (σ_t * s_t)^2 # squared deviations from real score
    scale = sum(scale, dims=1:(ndims(x_0)-1)) # L₂-norm

    return t, scale[:]
end

"""
    adapt_x!(x,
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
function adapt_x!(x,
                 forward_model::CliMAgen.VarianceExplodingSDE,
                 reverse_model::CliMAgen.VarianceExplodingSDE,
                 forward_t_end::FT,
                 reverse_t_end::FT) where{FT}
    _, forward_σ_end = CliMAgen.marginal_prob(forward_model, x, FT(forward_t_end)) # x only affects the mean, which we dont use
    _, reverse_σ_end = CliMAgen.marginal_prob(reverse_model, x, FT(reverse_t_end)) # x only affects the mean, which we dont use
    @. x = x * reverse_σ_end / forward_σ_end
end


"""
    make_icr(batch)

Computes and returns the mean condensation rate of the data `batch`.
"""
function make_icr(batch)
    τ = 1e-2 # condensation time scale which was set in the fluid simulations
    cond = @. batch * (batch > 0) / τ
    return  mean(cond, dims=(1,2))
end


### Return time computation for rare events
### Adapted from RareEvents.jl
"""
    block_applied_func(x::Vector, func::Function, m::Int)

Evaluates the function `func` over non-overlapping blocks
of length `m` (in indices) of the vector `x`.
"""
function block_applied_func(x::Vector, func::Function, m::Int)
    blocks = collect(Base.Iterators.partition(x, m))
    result = map(y-> func(y), blocks)
    output_length = Int(floor(length(x)/m))
    return result[1:output_length]
end

"""
     event_probability(a_m::Vector{FT},
                       lr::Vector{FT}
                       ) where {FT <:AbstractFloat}

Estimates the probability of an event as a function of magnitude,
including an estimate of the uncertainty of the probability.

The input arguments are: 
    (1) the `a_m` vector consisting of a scalar metric computed from M 
        uncorrelated trajectory segments;
    (2) a vector of likelihood ratios `lr`, equal to the ratio of the 
        target distribution to the importance distribution. This is used
        when the a_m are obtained using importance sampling; it is set 
        to a vector of 1 when using the results obtained from a direct 
        numerical simulation.

This estimates the probability using what I would predict
from a Poisson process, P(a) = M/q(a), where 
q is the observed number of segments with metric value
exceeding the threshold a. This computes the expected
number q(a) for an array of `a` values.

Under the original distribution p_0, we would have
q(a)/M = 1/M * ∑_i=1^M θ(a_i>a). If we sort the metric value
a_i in descending order, we can obtain a curve by noting that
q(a_j)/M = 1/M * ∑_i=1^j 1 = rank of a_j/M.

Using importance sampling: supposed we instead of M samples
from a distribution p_k. Then the expected number of events 
to exceed the threshold a under the model is given by
q(a)/M = 1/M * ∑_i=1^M θ(a_i>a) (p_0/p_k)_i, or
q(a_j)/M = 1/M * ∑_i=1^j (p_0/p_k)_i, if the a_j
are ranked (with their associated probablity ratios) 
in descending order.
"""
function event_probability(a_m::Vector{FT},
                           lr::Vector{FT}
                           ) where {FT<:AbstractFloat}
    sort_indices = reverse(sortperm(a_m))
    a_sorted = a_m[sort_indices]
    lr_sorted = lr[sort_indices] 
    M = length(a_m)
    # γa = P(X > a)
    γ = cumsum(lr_sorted)./M
    # Compute uncertainty 
    γ² = cumsum(lr_sorted.^2.0)./M
    σ_γ = sqrt.(γ² .-  γ.^2.0)/sqrt(M)
    return a_sorted, γ, σ_γ
end


"""
     return_curve(a_m::Vector{FT},
                  ΔT::FT,
                  lr::Vector{FT}
                  ) where {FT <:AbstractFloat}

Estimates the return time of an event as a function of magnitude,
returning an event magnitude array, a return time array, and
an estimate of the uncertainty.

The input arguments are: 
    (1) the `a_m` vector consisting of a scalar metric computed from M 
        uncorrelated trajectory segments;
    (2) the timespan ΔT  over which the metric is computed;
    (3) a vector of likelihood ratios `lr`, equal to the ratio of the 
        target distribution to the importance distribution. This is used
        when the a_m are obtained using importance sampling; it is set 
        to a vector of 1 when using the results obtained from a direct 
        numerical simulation.

This estimates the return curve using what I would predict
from a Poisson process, Return Time(a) = ΔT*M/q(a), where 
q is the observed number of segments with metric value
exceeding the threshold a. This computes the expected
number q(a) for an array of `a` values, and hence gets
an array of return times. This is slightly different from
what was reported in the Ragone et al. paper; we return both.

Under the original distribution p_0, we would have
q(a)/M = 1/M * ∑_i=1^M θ(a_i>a). If we sort the metric value
a_i in descending order, we can obtain a curve by noting that
q(a_j)/M = 1/M * ∑_i=1^j 1 = rank of a_j/M.

Using importance sampling: supposed we instead of M samples
from a distribution p_k. Then the expected number of events 
to exceed the threshold a under the model is given by
q(a)/M = 1/M * ∑_i=1^M θ(a_i>a) (p_0/p_k)_i, or
q(a_j)/M = 1/M * ∑_i=1^j (p_0/p_k)_i, if the a_j
are ranked (with their associated probablity ratios) 
in descending order.
"""
function return_curve(a_m::Vector{FT},
                      ΔT::FT,
                      lr::Vector{FT}
                      ) where {FT<:AbstractFloat}
    a_sorted, γ, σ_γ = event_probability(a_m, lr)
    
    # Compute return times rtn
    rtn_naive = ΔT ./  γ
    rtn_paper = -ΔT ./ log.(1.0 .- γ)

    # Compute uncertainty in rtn_naive
    σ_rtn = rtn_naive .* σ_γ./γ
    return a_sorted,rtn_naive, rtn_paper, σ_rtn
end

"""
    return_curve_plot(data::Vector{FT}, gen::Vector{FT}, ΔT::FT, savepath, plotname; logger=nothing) where {FT}

Creates and saves a plot of the return curve of the samples passed:
data, gen, both assumed to be of size nsamples. The
the plot is saved at joinpath(savepath, plotname).

The samples must be independent and associated with a time interval ΔT; this is
is required to turn the probability of the event into 
how often the event occurs.
"""
function return_curve_plot(data::Vector{FT}, gen::Vector{FT}, ΔT::FT, savepath, plotname; logger=nothing) where {FT}
    lr = ones(FT, length(gen))
    em, rtn_n, rtn_p, σ_rtn = return_curve(gen, ΔT, lr)
    Plots.plot(em, rtn_n,  ribbon = (σ_rtn, σ_rtn), label = "Generated", yaxis = :log10)
    lr = ones(FT, length(data))
    em, rtn_n, rtn_p, σ_rtn = return_curve(data, ΔT, lr)
    Plots.plot!(em, rtn_n,  ribbon = (σ_rtn, σ_rtn), label = "Training", ylabel = "Return time", xlabel = "Event magnitude", margin = 10Plots.mm)
    Plots.savefig(joinpath(savepath, plotname))
end


"""
    return_curve_plot(data::Vector{FT}, gen::Vector{FT}, ΔT::FT, savepath, plotname; logger=nothing) where {FT}

Creates and saves a plot of the return curve of the samples passed:
data, gen, both assumed to be of size nsamples. The
the plot is saved at joinpath(savepath, plotname).

The samples must be independent and associated with a time interval ΔT; this is
is required to turn the probability of the event into 
how often the event occurs.
"""
function event_probability_plot(train::Vector{FT}, gen::Vector{FT}, lr_gen, savepath, plotname; logger=nothing) where {FT}
    plts = []
    logscales = [true, false]
    for logscale in logscales
        em, γ, σ_γ = event_probability(gen, lr_gen)
        plt1 = Plots.plot()
        Plots.plot!(plt1, em, γ,  ribbon = (σ_γ, σ_γ), label = "Generated/IS", legend = :bottomleft)
        if logscale
         Plots.plot!(plt1, yaxis = :log10, ylim = [1e-6, 1])
        end

        N_gen = length(gen)
        lr_train = ones(FT, length(train))

        em, γ, σ_γ = event_probability(train[1:N_gen], lr_train[1:N_gen])
        Plots.plot!(plt1, em, γ,  ribbon = (σ_γ, σ_γ), label = "DS")

        em, γ, σ_γ = event_probability(train, lr_train)
        Plots.plot!(plt1, em, γ,  ribbon = (σ_γ, σ_γ), label = "Truth", ylabel = "Probability", xlabel = "Event magnitude", margin = 10Plots.mm)
        push!(plts, plt1)
    end
    plt = Plots.plot(plts..., layout = (1,2))
    Plots.savefig(plt, joinpath(savepath, plotname))
end
