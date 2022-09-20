"""
    ClimaGen.CliMAgen
"""
module CliMAgen

using ArgParse
using BSON
using CUDA
using DataLoaders
using Flux
using Functors
using Images
using MLDatasets
using MLUtils
using ProgressMeter
using Random
using Statistics
using Wandb

using CliMAgen

include("parameters.jl")
include("utils_data.jl")
include("utils_models.jl")
include("utils_training.jl")
include("models.jl")
include("networks.jl")
include("losses.jl")
include("optimizers.jl")
include("training.jl")

export struct2dict, dict2nt
export get_data_mnist, get_data_cifar10
export HyperParameters, parse_commandline
export VarianceExplodingSDE
export drift, diffusion, marginal_prob, score
export score_matching_loss
export NoiseConditionalScoreNetwork
export WarmupSchedule
export train!, load_model_and_optimizer, save_model_and_optimizer

end # module
