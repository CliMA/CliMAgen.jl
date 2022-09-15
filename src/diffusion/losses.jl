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
function score_matching_loss(model::AbstractDiffusionModel, x, ϵ=1.0f-5)
    # sample times approximate 𝔼[⋅] wrt. 𝘪 ∼ 𝒰(ϵ, 1)
    t = rand!(similar(x, size(x)[end])) .* (1 - ϵ) .+ ϵ

    # sample from normal prior to approximate 𝔼[⋅] wrt. 𝘹(0) ∼ 𝒫₀(𝘹)
    σ_0 = sigma(model, 0)
    x_0 = σ_0 .* randn!(similar(x))

    # sample from normal marginal to approximate 𝔼[⋅] wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0))
    σ_t = expand_dims(sigma(model, t), ndims(x) - 1)
    x_t = x .+ σ_t .* x_0

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = model.score(x_t, t)

    # Assume that λ(t) = σ(t)² and pull it into L₂-norm
    loss = @. (x_0 + s_t)^2 # squared deviations from real score
    loss = sum(loss, dims=1:(ndims(x)-1)) # L₂-norm
    loss = Statistics.mean(loss) # mean over samples/batches

    return loss
end