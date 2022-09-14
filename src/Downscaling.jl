module Downscaling

using Flux
using Flux.Optimise: update!
import Flux.Optimise: apply!
using Functors: @functor
using Statistics

include("generators_2d.jl")
include("discriminators_2d.jl")
include("generators_1d.jl")
include("discriminators_1d.jl")
include("optimizers.jl")

export ConvBlock1D
export ResidualBlock1D
export PatchBlock1D
export PatchDiscriminator1D
export UNetGenerator1D

export ConvBlock2D
export ResidualBlock2D
export PatchBlock2D
export PatchDiscriminator2D
export UNetGenerator2D
export NoisyUNetGenerator1D
export NoisyUNetGenerator2D

export WarmupSchedule, create_optimizer, OptimizerHyperParams
end
