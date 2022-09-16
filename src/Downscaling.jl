"""
    ClimaGen.Downscaling
Conventions:
- bla
- blup
Cool story about what we are doing.
"""
module Downscaling

using Flux
using Functors
import Flux.Optimise: apply!
using Random
using Statistics

include("utils_models.jl")

include("models.jl")
include("losses.jl")
include("networks.jl")
include("optimizers.jl")

export AbstractDiffusionModel
export VarianceExplodingSDE
export drift, sigma, diffusion, score
export score_matching_loss
export NoiseConditionalScoreNetwork
export WarmupSchedule, create_optimizer, OptimizerHyperParams
end # module
