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
using Random
using Statistics

include("utils_models.jl")

include("models.jl")
include("losses.jl")
include("networks.jl")

export AbstractDiffusionModel
export VarianceExplodingSDE
export drift, sigma, diffusion, score
export score_matching_loss
export NoiseConditionalScoreNetwork

end # module