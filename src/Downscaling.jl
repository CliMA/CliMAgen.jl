module Downscaling

using Flux
using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Functors: @functor
using Statistics
using Zygote

include("generators.jl")
include("discriminators.jl")
include("generators_1d.jl")
include("discriminators_1d.jl")
include("op_generators.jl")

export ConvBlock
export ResidualBlock
export PatchBlock
export PatchDiscriminator
export UNetGenerator
export UNetGeneratorAR
export NoisyUNetGenerator

export ConvBlock1D
export ResidualBlock1D
export PatchBlock1D
export PatchDiscriminator1D
export UNetGenerator1D

export OperatorBlock2D
export UNetOperator2D
export UNetOperatorGenerator
export UNetOperatorDiscriminator

end
