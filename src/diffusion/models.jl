abstract type AbstractDiffusionModel end

"""
    ClimaGen.drift
"""
function drift end

"""
    ClimaGen.sigma
"""
function sigma end

"""
    ClimaGen.diffusion
"""
function diffusion end

"""
    ClimaGen.score
"""
function score end

"""
    ClimaGen.forward_sde
"""
function forward_sde(m::AbstractDiffusionModel)
    f(x, t) = ones(eltype(x), size(x)) .* drift(m, t)
    g(x, t) = ones(eltype(x), size(x)) .* diffusion(m, t)
    return f, g
end

"""
    ClimaGen.reverse_sde
"""
function reverse_sde(m::AbstractDiffusionModel)
    f(x, t) = drift(m, t) .- diffusion(m, t)^2 .* score(m, x, t)
    g(x, t) = ones(eltype(x), size(x)) .* diffusion(m, t)
    return f, g
end

"""
    ClimaGen.reverse_ode
"""
function reverse_ode(m::AbstractDiffusionModel)
    f(x, t) = drift(m, t) .- diffusion(m, t)^2 .* score(m, x, t) ./ 2
    return f
end

struct VarianceExplodingSDE{FT,S} <: AbstractDiffusionModel
    σ_min::FT
    σ_max::FT
    score::Function
end
"""
    ClimaGen.VarianceExplodingSDE
"""
function VarianceExplodingSDE(;
    σ_min::FT=0.01f0,
    σ_max::FT=50.0f0,
    score::S=identity
) where {FT,S<:Function}
    return VarianceExplodingSDE{FT,S}(σ_min, σ_max, (x, t) -> score(x))
end

function t_end(::VarianceExplodingSDE{FT}) where {FT}
    return one(FT)
end

function drift(::VarianceExplodingSDE{FT}, t) where {FT}
    return zeros(FT, size(t))
end

function sigma(m::VarianceExplodingSDE, t)
    return @. m.σ_min * (m.σ_max / m.σ_min)^t
end

function diffusion(m::VarianceExplodingSDE, t)
    return sigma(m, t) * sqrt(2 * (log(m.σ_max) - log(m.σ_min)))
end

function score(m::VarianceExplodingSDE, x, t)
    return m.score(x, t)
end