module Downscaling

using Flux
using Functors: @functor

include("generators.jl")
include("discriminators.jl")

export ConvBlock
export ResidualBlock
export PatchBlock
export PatchDiscriminator
export UNetGenerator

end
