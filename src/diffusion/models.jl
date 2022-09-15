abstract type AbstractDiffusionModel end

"""
    ClimaGen.drift
"""
function drift end

"""
    ClimaGen.diffusion
"""
function diffusion end


"""
    ClimaGen.marginal_prob
"""
function marginal_prob end

"""
    ClimaGen.score
"""
function score(m::AbstractDiffusionModel, x, t)
    _, σ_t = marginal_prob(m, x, t)
    return m.net(x, t) ./ σ_t
end

"""
    ClimaGen.forward_sde
"""
function forward_sde(m::AbstractDiffusionModel)
    f(x, p, t) = ones(eltype(x), size(x)) .* drift(m, t)
    g(x, p, t) = ones(eltype(x), size(x)) .* diffusion(m, t)
    return f, g
end

"""
    ClimaGen.reverse_sde
"""
function reverse_sde(m::AbstractDiffusionModel)
    f(x, p, t) = drift(m, t) .- diffusion(m, t) .^ 2 .* score(m, x, t)
    g(x, p, t) = ones(eltype(x), size(x)) .* diffusion(m, t)
    return f, g
end

"""
    ClimaGen.reverse_ode
"""
function reverse_ode(m::AbstractDiffusionModel)
    f(x, p, t) = drift(m, t) .- diffusion(m, t) .^ 2 .* score(m, x, t) ./ 2
    return f
end

struct VarianceExplodingSDE{FT,N} <: AbstractDiffusionModel
    σ_min::FT
    σ_max::FT
    net::N
end

# only the neural network is trainable within the diffusion model
Flux.params(m::AbstractDiffusionModel) = Flux.params(m.net)

"""
    ClimaGen.VarianceExplodingSDE
"""
function VarianceExplodingSDE(;
    σ_min::FT=0.01f0,
    σ_max::FT=50.0f0,
    net::N
) where {FT,N}
    return VarianceExplodingSDE{FT,N}(σ_min, σ_max, net)
end

@functor VarianceExplodingSDE

function drift(::VarianceExplodingSDE{FT}, t) where {FT}
    return zeros(FT, size(t))
end

function diffusion(m::VarianceExplodingSDE, t)
    std = @. m.σ_min * (m.σ_max / m.σ_min)^t
    return @. std * sqrt(2 * (log(m.σ_max) - log(m.σ_min)))
end

function marginal_prob(m::VarianceExplodingSDE, x_0, t)
    μ_t = x_0
    σ_t = @. m.σ_min * (m.σ_max / m.σ_min)^t
    return μ_t, expand_dims(σ_t, ndims(μ_t) - 1)
end