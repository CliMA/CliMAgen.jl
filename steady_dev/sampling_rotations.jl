using LinearAlgebra
using CairoMakie
using Statistics
using ProgressBars
using Flux
using CliMAgen
using BSON
using HDF5
using CUDA
using Random

include("sampler.jl")


device = Flux.gpu

function load_model(checkpoint_path)
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    score_model_smooth = device(model_smooth)
    return score_model_smooth
end

model = "checkpoint_conditional_rotations_steady_online_timestep_200000.bson"

function GaussianFourierProjection(embed_dim::Int, embed_dim2::Int, scale::FT) where {FT}
    Random.seed!(1234) # same thing every time
    W = randn(FT, embed_dim ÷ 2, embed_dim2) .* scale
    return GaussianFourierProjection(W)
end

gfp = GaussianFourierProjection(128, 64, 30.0f0)

score_model_smooth_s = load_model(model)

nsamples = 8
nsteps = 250
inchannels = 1
Ny = 64
resolution = (128, 64)
time_steps, Δt, init_x = setup_sampler(
    score_model_smooth_s,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)
rng = MersenneTwister(1234)
ĉ = reshape(gfp(0), 128, 64, 1, 1)
c = zeros(Float32, 128, 64, 1, nsamples)
c .= ĉ
samples = Array(Euler_Maruyama_sampler(score_model_smooth, init_x, time_steps, Δt; rng, c))