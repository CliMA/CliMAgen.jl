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
    
    These functions expect the input `t` to be a scalar, in order to work with DifferentialEquations.
"""
function forward_sde(m::AbstractDiffusionModel)
    function f(x, p, t)
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(drift(m, t), ndims(x) - 1)
    end
    function g(x, p, t)
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(diffusion(m, t), ndims(x) - 1)
    end

    return f, g
end

"""
    ClimaGen.reverse_sde

    These functions expect the input `t` to be a scalar, in order to work with DifferentialEquations.
"""
function reverse_sde(m::AbstractDiffusionModel)
    function f(x, p, t) 
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(drift(m, t), ndims(x) - 1) .- expand_dims(diffusion(m, t), ndims(x) - 1) .^ 2 .* score(m, x, t)
    end
    function g(x, p, t)
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(diffusion(m, t), ndims(x) - 1)
    end
    return f, g
end

"""
    ClimaGen.probability_flow_ode

    These functions expect the input `t` to be a scalar, in order to work with DifferentialEquations.
"""
function probability_flow_ode(m::AbstractDiffusionModel)
    function f(x, p, t) 
        t = fill!(similar(x, size(x)[end]), 1) .* t
        expand_dims(drift(m, t), ndims(x) - 1) .- expand_dims(diffusion(m, t), ndims(x) - 1) .^ 2 .* score(m, x, t) ./ 2
    end
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

@functor VarianceExplodingSDE

function drift(::VarianceExplodingSDE{FT}, t) where {FT}
    return similar(t) .* FT(0)
end

function diffusion(m::VarianceExplodingSDE, t)
    return @. m.σ_min * (m.σ_max/m.σ_min)^t*sqrt(2*log(m.σ_max/m.σ_min))
end

function marginal_prob(m::VarianceExplodingSDE, x_0, t)
    μ_t = x_0
    σ_t = @. m.σ_min * (m.σ_max/m.σ_min)^(t)
    return μ_t, expand_dims(σ_t, ndims(μ_t) - 1)
end


"""
    ClimaGen.VarianceExplodingSDEVariant
"""
Base.@kwdef struct VarianceExplodingSDEVariant{FT,N} <: AbstractDiffusionModel
    σ_max::FT
    σ_min::FT
    n::FT
    net::N
end

@functor VarianceExplodingSDEVariant

function drift(::VarianceExplodingSDEVariant{FT}, t) where {FT}
    return similar(t) .* FT(0)
end

function diffusion(m::VarianceExplodingSDEVariant, t)
    tn = t.^m.n
    tnm1 = t.^(m.n-1)
    return @. sqrt(2*(m.σ_max - m.σ_min) * m.n * tnm1 * m.σ_min * ((m.σ_max/m.σ_min - 1) * tn + 1))
end

function marginal_prob(m::VarianceExplodingSDEVariant, x_0, t)
    μ_t = x_0
    tn = t.^m.n
    σ_t = @. m.σ_min * ((m.σ_max/m.σ_min - 1) * tn + 1)
    return μ_t, expand_dims(σ_t, ndims(μ_t) - 1)
end
