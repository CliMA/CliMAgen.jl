"""
    ClimaGen.CliMAgen
"""
module CliMAgen

using ArgParse
using BSON
using CUDA
using Flux
using Functors
using Logging
using ProgressMeter
using Random
using Statistics

using CliMAgen

include("utils.jl")
include("logging.jl")
include("parameters.jl")
include("models.jl")
include("networks.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")
include("sampling.jl")

export struct2dict, dict2nt, parse_commandline
export HyperParameters
export VarianceExplodingSDE
export drift, diffusion, marginal_prob, score
export score_matching_loss
export NoiseConditionalScoreNetwork
export WarmupSchedule, ExponentialMovingAverage
export train!, load_model_and_optimizer, save_model_and_optimizer
export setup_sampler, Euler_Maruyama_sampler, predictor_corrector_sampler

include("networks_new.jl")
export ConvMixer

end # module
