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
    s_t = score(model, x_t, t)

    # split into spatial averages and deviations
    nspatial = ndims(x_0)-2
    n = prod(size(z)[1:nspatial])
    √n = convert(eltype(x_0), sqrt(n))
    z_avg = Statistics.mean(z, dims=1:nspatial)
    z_dev = @. z - z_avg
    z_star = @. √n * z_avg
    s_t_avg = Statistics.mean(s_t, dims=1:nspatial)
    s_t_dev = @. s_t - s_t_avg

    # spatial average component of loss
    # We have retained the 1/n in front of the avg loss in order to compare with original net
    loss_avg = @. (z_star + √n * σ_t * s_t_avg)^2 # squared deviations from real score
    loss_avg = mean(loss_avg, dims=1:(ndims(x_0)-1)) # spatial & channel mean 
    loss_avg = 1/n * Statistics.mean(loss_avg) # mean over samples/batches

    # spatial deviation component of loss
    dev = @. (z_dev + σ_t * s_t_dev)
    loss_dev = dev.^2 # squared deviations from real score
    loss_dev = mean(loss_dev, dims=1:(ndims(x_0)-1)) # spatial & channel mean 
    loss_dev = Statistics.mean(loss_dev) # mean over samples/batches    

    # spectral loss for deviations
    dim = size(dev, 1)
    dev_fft = abs.(fft(dev, 1:(ndims(x_0)-2))) ./ dim^2
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
    r = zeros(eltype(x_0), size(rx, 1), size(ry, 1))
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        r[i, j] = sqrt(R_x[i]^2 + R_y[j]^2)
    end
    k = range(1, stop=k_nyq, step=1)
    endk = size(k, 1)
    contribution = zeros(eltype(x_0),(endk, (size(x_0)[ndims(x_0)-1],size(x_0)[end])...))
    spectrum = zeros(eltype(x_0),(endk, (size(x_0)[ndims(x_0)-1],size(x_0)[end])...))
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
    loss_spec = Statistics.mean(spectrum) # avg over channels and batch members
    return [loss_avg, loss_dev, loss_spec]
end




