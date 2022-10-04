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

# only the neural network is trainable within the diffusion model
Flux.params(m::AbstractDiffusionModel) = Flux.params(m.net)

Base.deepcopy(m::M) where {M <: AbstractDiffusionModel} = 
    M((deepcopy(getfield(m, f)) for f in fieldnames(M))...)

"""
    ClimaGen.VarianceExplodingSDE
"""
Base.@kwdef struct VarianceExplodingSDE{FT,N} <: AbstractDiffusionModel
    σ_max::FT
    σ_min::FT
    net::N
end
function VarianceExplodingSDE(hpmodel::NamedTuple; net)
    return VarianceExplodingSDE(hpmodel.σ_max, hpmodel.σ_min, net)
end

@functor VarianceExplodingSDE

function drift(::VarianceExplodingSDE{FT}, t) where {FT}
    return zeros(FT, size(t))
end

function diffusion(m::VarianceExplodingSDE, t)
    return @. m.σ_min * (m.σ_max/m.σ_min)^t*sqrt(2*log(m.σ_max/m.σ_min))
end

function marginal_prob(m::VarianceExplodingSDE, x_0, t)
    μ_t = x_0
    σ_t = @. m.σ_min * (m.σ_max/m.σ_min)^t
    return μ_t, expand_dims(σ_t, ndims(μ_t) - 1)
end
