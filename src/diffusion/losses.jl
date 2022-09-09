
"""
Model loss following the denoising score matching objectives:

# Notes
Denoising score matching objective:
```julia
min wrt. θ (
    𝔼 wrt. 𝘵 ∼ 𝒰(0, 𝘛)[
        λ(𝘵) * 𝔼 wrt. 𝘹(0) ∼ 𝒫₀(𝘹) [
            𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0)) [
                (||𝘚₀(𝘹(𝘵), 𝘵) - ∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))] ||₂)²
            ]
        ]
    ]
)
``` 
Where 𝒫₀ₜ(𝘹(𝘵) | 𝘹(0)) and λ(𝘵), are available analytically and
𝘚₀(𝘹(𝘵), 𝘵) is estimated by a U-Net architecture.

# References:
http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf \n
https://yang-song.github.io/blog/2021/score/#estimating-the-reverse-sde-with-score-based-models-and-score-matching \n
https://yang-song.github.io/blog/2019/ssm/
"""
function model_loss(model::AbstractDiffusionModel, x, ϵ=1.0f-5)
    # sample times approximate 𝔼[⋅] wrt. 𝘪 ∼ 𝒰(ϵ, 1)
    t = rand!(similar(x, size(x)[end])) .* (1.0f0 - ϵ) .+ ϵ

    # sample from prior to approximate 𝔼[⋅] wrt. 𝘹(0) ∼ 𝒫₀(𝘹)
    μ_0, σ_0 = prior(model)
    x_0 = μ_0 .+ σ_0 .* randn!(similar(x))
    
    # sample from marginal to approximate 𝔼 wrt. 𝘹(t) ∼ 𝒫₀ₜ(𝘹(𝘵)|𝘹(0))
    μ_t, σ_t = marginal(model, x, t)
    x_t = μ_t .+ σ_t .* x_0

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = model.score(x_t, t)

    # L₂ norm over WHC dimensions and mean over batches
    # Assume that λ(t) = σ(t)²  and pulled it into L₂-norm
    loss = @. (x_0 + σ_t^2 * s_t)^2 # squared deviations from real score
    loss = sum(loss, dims=1:(ndims(x)-1)) # L₂-norm
    loss = mean(loss) # mean over samples/batches

    return loss
end