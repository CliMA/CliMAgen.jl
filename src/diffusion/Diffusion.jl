"""
    ClimaGen.DiffusionModels
Conventions:
- bla
- blup
Cool story about what we are doing.
"""
module DiffusionModels

using Flux
using Functors
using Statistics: mean

include("models.jl")
include("losses.jl")
include("networks.jl")

export NCNN
export VarianceExplodingSDE, VESDE
export drift, diffusion, prior, marginal, score
export score_matching_loss

end # module