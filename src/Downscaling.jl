module Downscaling

using Flux
using Functors: @functor

#include("generators.jl")
include("discriminators.jl")

export PatchBlock
export PatchDiscriminator

end
