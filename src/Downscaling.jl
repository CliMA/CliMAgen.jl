module Downscaling

using Flux
using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Functors: @functor
using Statistics
using Zygote

include("op_generators.jl")
include("generators.jl")
include("discriminators.jl")

export ConvBlock
export ResidualBlock
export OperatorConvBlock
export OperatorResidualBlock
export PatchBlock
export PatchDiscriminator
export UNetGenerator
export OperatorUNetGenerator
export CycleGAN
export update_cyclegan

end
