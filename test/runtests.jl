using CliMAgen
using CUDA
using Flux
using Test
using Random
using Statistics: mean

FT = Float32

include("./tests_models.jl")
include("./test_networks.jl")
include("./tests_optimizers.jl")
include("./tests_utils.jl")
