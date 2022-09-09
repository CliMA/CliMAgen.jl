
"""
    ClimaGen.DiffusionModels
Conventions:
- bla
- blup
Cool story about what we are doing.
"""
module DiffusionModels

using Statistics: mean
using Distributions: MvNormal

abstract type AbstractDiffusionModel end

function drift end
function diffusion end
function marginal end
function prior end
function score end
function forward_sde end
function backward_sde end
function reverse_ode end

Base.@kwdef struct VarianceExplodingSDE{FT,S} <: AbstractDiffusionModel
    σ_min::FT = 0.01f0
    σ_max::FT = 50.0f0
    score::S = Function
end

const VESDE = VarianceExplodingSDE

function drift(::VESDE, t) 
    return 0
end

function diffusion(m::VESDE, t)
    return @. m.σ_min / (m.σ_max / m.σ_min)^t * sqrt(2 * (log(m.σ_max) - log(m.σ_max)))
end

function marginal(m::VESDE, μ, t)
    mean = μ
    std = diffusion.(Ref(m), t) ./ sqrt(2 * (log(m.σ_max) - log(m.σ_max)))
    return mean, std
end

function prior(m::VESDE)
    mean = 0
    std = m.σ
    return mean, std
end

function score(m::VESDE, x, t)
    return m.score(x, t)
end

function forward_sde(m::AbstractDiffusionModel)
    f(x, t) = drift(m, t)
    g(x, t) = diffusion(m, t)
    return f, g
end

function reverse_sde(m::AbstractDiffusionModel)
    f(x, t) = drift(m, t) .- diffusion(m, t)^2 .* score(m, x, t)
    g(x, t) = diffusion(m, t)
    return f, g
end

function reverse_ode(m::AbstractDiffusionModel)
    f(x, t) = drift(m, t) .- diffusion(m, t)^2 .* score(m, x, t) ./ 2
    return f
end

end # module