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

Base.@kwdef struct VarianceExplodingSDEParams{FT} <: AbstractModelParams{FT}
    σ_max::FT = 10.0
end

struct VarianceExplodingSDE{FT,N} <: AbstractDiffusionModel
    σ_max::FT
    net::N
end

# only the neural network is trainable within the diffusion model
Flux.params(m::AbstractDiffusionModel) = Flux.params(m.net)

"""
    ClimaGen.VarianceExplodingSDE
"""
function VarianceExplodingSDE(;
    hpmodel::VarianceExplodingSDEParams{FT},
    net::N
) where {FT,N}
    return VarianceExplodingSDE{FT,N}(hpmodel.σ_max, net)
end

@functor VarianceExplodingSDE

function drift(::VarianceExplodingSDE{FT}, t) where {FT}
    return zeros(FT, size(t))
end

function diffusion(m::VarianceExplodingSDE, t)
    return @. m.σ_max^t
end

function marginal_prob(m::VarianceExplodingSDE, x_0, t)
    μ_t = x_0
    σ_t = @. sqrt((m.σ_max^(2*t)-1)/2/log(m.σ_max))
    return μ_t, expand_dims(σ_t, ndims(μ_t) - 1)
end
