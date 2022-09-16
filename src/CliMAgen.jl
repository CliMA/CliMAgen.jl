"""
    ClimaGen.CliMAgen
Conventions:
- bla
- blup
Cool story about what we are doing.
"""
module CliMAgen

using Flux
using Functors
import Flux.Optimise: apply!
using Random
using Statistics

include("parameters.jl")
include("utils_models.jl")
include("models.jl")
include("losses.jl")
include("networks.jl")
include("optimizers.jl")

export AbstractDiffusionModel
export VarianceExplodingSDE,VarianceExplodingSDEParams
export drift, sigma, diffusion, score
export score_matching_loss
export NoiseConditionalScoreNetwork
export WarmupSchedule, create_optimizer, AdamOptimizerParams
export TrainParams, DataParams, Parameters, Args
end # module
