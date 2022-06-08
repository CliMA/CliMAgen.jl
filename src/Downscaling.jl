module Downscaling

using Flux
using Flux.Optimise: update!
using Flux.Losses: mse, mae
using Functors: @functor
using Statistics
using Zygote

include("generators.jl")
include("discriminators.jl")
include("Trainer.jl")

#include("cyclegans.jl")

export ConvBlock
export ResidualBlock
export PatchBlock
export PatchDiscriminator
export UNetGenerator
export CycleGAN
export update_cyclegan

end
