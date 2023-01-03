"""
    ClimaGen.score_matching_loss

# Notes
Denoising score matching objective:
```julia
min wrt. θ (
    𝔼 wrt. 𝘵 ∼ 𝒰(0, 𝘛)[
        λ(𝘵) * 𝔼 wrt. 𝘹(0) ∼ 𝒫₀(𝘹) [
            𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0)) [
                (||s₀(𝘹(𝘵), 𝘵) - ∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))] ||₂)²
            ]
        ]
    ]
)
``` 
Where 𝒫₀ₜ(𝘹(𝘵) | 𝘹(0)) and λ(𝘵), are available analytically and
s₀(𝘹(𝘵), 𝘵) is estimated by a U-Net architecture.

# References:
https://arxiv.org/abs/2011.13456
https://arxiv.org/abs/1907.05600
"""
function score_matching_loss(model::AbstractDiffusionModel, x_0, ϵ=1.0f-5)
    # sample times
    t = rand!(similar(x_0, size(x_0)[end])) .* (1 - ϵ) .+ ϵ

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = score(model, x_t, t)

    # Assume that λ(t) = σ(t)² and pull it into L₂-norm
    # Below, z / σ_t = -∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))
    loss = @. (z + σ_t * s_t)^2 # squared deviations from real score
    loss = Statistics.mean(loss, dims=1:(ndims(x_0)-1)) # L₂-norm
    loss = Statistics.mean(loss) # mean over samples/batches

    return loss
end

function score_matching_loss_variant(model::AbstractDiffusionModel, x_0, ϵ=1.0f-5)
    # sample times
    t = rand!(similar(x_0, size(x_0)[end])) .* (1 - ϵ) .+ ϵ

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = CliMAgen.score(model, x_t, t)

    # split into spatial averages and deviations
    nspatial = ndims(x_0)-2
    n = prod(size(z)[1:nspatial])
    sqrtn = convert(eltype(x_0), sqrt(n))
    z_avg = Statistics.mean(z, dims=1:nspatial)
    z_dev = @. z - z_avg
    z_star = @. sqrtn * z_avg
    s_t_avg = Statistics.mean(s_t, dims=1:nspatial)
    s_t_dev = @. s_t - s_t_avg

    # spatial average component of loss
    # We have retained the 1/n in front of the avg loss in order to compare with original net
    loss_avg = @. (z_star + sqrtn * σ_t * s_t_avg)^2 # squared deviations from real score
    loss_avg = mean(loss_avg, dims=1:(ndims(x_0)-1)) # spatial & channel mean 
    loss_avg = 1/n * Statistics.mean(loss_avg) # mean over samples/batches

    # spatial deviation component of loss
    weight = expand_dims((2 .-t), ndims(x_t) - 1)
    loss_dev = @. (z_dev + σ_t * s_t_dev)^2 * weight # squared deviations from real score
    loss_dev = mean(loss_dev, dims=1:(ndims(x_0)-1)) # spatial & channel mean 
    loss_dev = Statistics.mean(loss_dev) # mean over samples/batches    

    # spectral loss for deviations - this does not improve
    # dev = @. (z_dev + σ_t * s_t_dev)
  #  dim = size(dev, 1)
  #  dev_fft = abs2.(FFTW.fft(dev, 1:(ndims(x_0)-2))) ./ dim^2
  #  loss_spec = Statistics.mean(dev_fft) # avg over channels and batch members
    return [loss_avg, loss_dev]
end

# does not work/scalar indexing on GPU
# dev_fft = abs.(FFTW.fft(dev, 1:(ndims(x_0)-2))) ./ dim^2
#spectrum = azimuthal_average_ps(dev_fft, dim)
#loss_spec = Statistics.mean(spectrum) # avg over
function azimuthal_average_ps(dev_fft::CuArray, dim::Int)
    if mod(dim, 2) == 0 
        rx = CUDA.range(0, stop=dim - 1, step=1) .- dim / 2 .+ 1
        ry = CUDA.range(0, stop=dim - 1, step=1) .- dim / 2 .+ 1
        R_x = circshift(rx', (1, dim / 2 + 1))
        R_y = circshift(ry', (1, dim / 2 + 1))
        k_nyq = dim / 2
    else
        rx = CUDA.range(0, stop=dim - 1, step=1) .- (dim - 1) / 2
        ry = CUDA.range(0, stop=dim - 1, step=1) .- (dim - 1) / 2
        R_x = circshift(rx', (1, (dim + 1) / 2))
        R_y = circshift(ry', (1, (dim + 1) / 2))
        k_nyq = (dim - 1) / 2
    end
    r = CUDA.zeros(eltype(dev_fft), size(rx, 1), size(ry, 1))
    k = CUDA.range(1, stop=k_nyq, step=1)
    endk = size(k, 1)
    contribution = CUDA.zeros(eltype(dev_fft),(endk, (size(dev_fft)[ndims(dev_fft)-1],size(dev_fft)[end])...))
    spectrum = CUDA.zeros(eltype(dev_fft),(endk, (size(dev_fft)[ndims(dev_fft)-1],size(dev_fft)[end])...))

    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        r[i, j] = sqrt(R_x[i]^2 + R_y[j]^2)
    end

    for N in 2:Int64(k_nyq - 1)
        for i in 1:size(rx, 1), j in 1:size(ry, 1)
            if (r[i, j] <= (k'[N+1] + k'[N]) / 2) &&
            (r[i, j] > (k'[N] + k'[N-1]) / 2)
                spectrum[N,:,:] +=  dev_fft[i, j,:,:].^2
                contribution[N,:,:] .+=  1
            end
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j]  <= (k'[2] + k'[1]) / 2)
            spectrum[1,:,:] += dev_fft[i, j,:,:].^2
            contribution[1,:,:] .+=  1
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j]  <= k'[endk]) &&
            (r[i, j]  > (k'[endk] + k'[endk-1]) / 2)
            spectrum[endk,:,:] +=  dev_fft[i, j,:,:].^2
            contribution[endk,:,:] .+= 1
        end
    end
    spectrum = @. (spectrum * 2 * pi * k ^ 2 / contribution)
    return spectrum
end


function azimuthal_average_ps(dev_fft::Array, dim::Int)
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
    r = zeros(eltype(dev_fft), size(rx, 1), size(ry, 1))
    k = range(1, stop=k_nyq, step=1)
    endk = size(k, 1)
    contribution = zeros(eltype(dev_fft),(endk, (size(dev_fft)[ndims(dev_fft)-1],size(dev_fft)[end])...))
    spectrum = zeros(eltype(dev_fft),(endk, (size(dev_fft)[ndims(dev_fft)-1],size(dev_fft)[end])...))

    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        r[i, j] = sqrt(R_x[i]^2 + R_y[j]^2)
    end

    for N in 2:Int64(k_nyq - 1)
        for i in 1:size(rx, 1), j in 1:size(ry, 1)
            if (r[i, j] <= (k'[N+1] + k'[N]) / 2) &&
            (r[i, j] > (k'[N] + k'[N-1]) / 2)
                spectrum[N,:,:] +=  dev_fft[i, j,:,:].^2
                contribution[N,:,:] .+=  1
            end
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j]  <= (k'[2] + k'[1]) / 2)
            spectrum[1,:,:] += dev_fft[i, j,:,:].^2
            contribution[1,:,:] .+=  1
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j]  <= k'[endk]) &&
            (r[i, j]  > (k'[endk] + k'[endk-1]) / 2)
            spectrum[endk,:,:] +=  dev_fft[i, j,:,:].^2
            contribution[endk,:,:] .+= 1
        end
    end
    spectrum = @. (spectrum * 2 * pi * k ^ 2 / contribution)
    return spectrum
end
