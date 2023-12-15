### Code Copied from https://github.com/tk3369/YeoJohnsonTrans.jl on 12/15/2023
### Authored by github user tk3369
### If we want to include, we should ask the author to register the package?
module YeoJohnsonTransform

using Optim: optimize, minimizer
using Statistics: mean, var
using StatsBase: geomean

"""
    transform(x)

Transform an array using Yeo-Johnson method.  The power parameter λ is derived
from maximizing a log-likelihood estimator. 
"""
function transform(x; optim_args...)
    λ, details = lambda(x; optim_args...)
    #@info "estimated lambda = $λ"
    transform(x, λ)
end

"""
    transform(x, λ)

Transform an array using Yeo-Johnson method with the provided power parameter λ. 
"""
function transform(x, λ) 
    x′ = similar(x, Float64)
    for (i, x) in enumerate(x)
        if x >= 0
            x′[i] = λ ≈ 0 ? log(x + 1) : ((x + 1)^λ - 1)/λ 
        else
            x′[i] = λ ≈ 2 ? -log(-x + 1) : -((-x + 1)^(2 - λ) - 1) / (2 - λ)
        end
    end
    x′
end

"""
    inversetransform(x, λ)

Inverse transforms an array using Yeo-Johnson method with the provided power parameter λ. 
"""
function inversetransform(x, λ) 
    x′ = similar(x, Float64)
    for (i, x) in enumerate(x)
        if x >= 0
            x′[i] = λ ≈ 0 ? exp(x) - 1 : (λ*x+1)^(1/λ)-1 
        else
            x′[i] = λ ≈ 2 ? 1 - exp(-x) : 1 - (-(2-λ)*x + 1)^(1/(2-λ))
        end
    end
    x′
end

"""
    lambda(x; interval = (-2.0, 2.0), optim_args...)

Calculate lambda from an array using a log-likelihood estimator.

Keyword arguments:
- interval: search interval
- optim_args: keyword arguments accepted by Optim.optimize function

See also: [`log_likelihood`](@ref)
"""
function lambda(x; interval = (-4.0, 4.0), optim_args...)
    i1, i2 = interval
    res = optimize(λ -> -log_likelihood(x, λ), i1, i2; optim_args...)
    (value=minimizer(res), details=res)
end

"""
    log_likelihood(x, λ)

Return log-likelihood for the given array and lambda.
"""
function log_likelihood(x, λ)
    N = length(x)
    𝐲 = transform(float.(x), λ)
    σ² = var(𝐲, corrected = false)
    c = sum(sign.(x) .* log.(abs.(x) .+ 1))
    llf = -N / 2.0 * log(σ²) + (λ - 1) * c
    #@info "λ = $λ => σ²=$σ², c=$c, llf=$llf"
    llf
end

end # module