using LinearAlgebra
using Statistics
using ProgressBars
using Flux
using CliMAgen
using BSON
using HDF5
using CUDA
using Random
import CliMAgen: GaussianFourierProjection
include("../sandbox/sampler.jl")

function load_model(checkpoint_path)
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    device = Flux.gpu
    score_model_smooth = device(model_smooth)
    return score_model_smooth
end

function GaussianFourierProjection(embed_dim::Int, embed_dim2::Int, scale::FT) where {FT}
    Random.seed!(1234) # same thing every time
    W = randn(FT, embed_dim ÷ 2, embed_dim2) .* scale
    return GaussianFourierProjection(W)
end

function generate_rotation_samples(rotation_index, nsamples, nbatches, batchsize)
    nsteps = 250
    inchannels = 1
    Nx = 128
    Ny = 64
    resolution = (Nx, Ny)
    FT = Float32
    samples = zeros(FT, (resolution...,inchannels, nsamples))
    c = zeros(Float32, Nx, Ny, 1, batchsize)
    device = Flux.gpu
    model = "checkpoint_capacity_conditional_rotations_steady_online_timestep_1000000.bson"

    gfp = GaussianFourierProjection(Nx, Ny, 30.0f0)
    @info "loading model"
    score_model_smooth_s = load_model(model)

    @info "setting up sampling"
    rotation_rates = [Float32(0.6e-4), Float32(1.1e-4), Float32(1.5e-4), Float32(7.29e-5)]
    a = 5e-5 
    b = 1e-4
    times = (rotation_rates .- a) ./ (b-a)

    c .= reshape(gfp(times[rotation_index]), Nx, Ny, 1, 1)
    @info "sampling $(rotation_rates[rotation_index])"
    for batch in 1:nbatches
        @info batch
        time_steps, Δt, init_x = setup_sampler(score_model_smooth_s,
                                                device,
                                                resolution,
                                                inchannels;
                                                num_images=batchsize,
                                                num_steps=nsteps,
                                            )
        rng = MersenneTwister(batch*rotation_index)
        samples[:,:,:,(batch-1)*batchsize+1:batch*batchsize] .= Array(Euler_Maruyama_sampler(score_model_smooth_s, init_x, time_steps, Δt; rng, c))
    end

    hfile = h5open("analysis/rotation_rate_samples_$(rotation_index).hdf5", "w")
    hfile["samples"] = samples
    hfile["rotation"] = rotation_rates[rotation_index]
    close(hfile)
end

function main(rotation_index)
    nsamples = 1000
    nbatches = 4
    batchsize = 250
    generate_rotation_samples(rotation_index, nsamples, nbatches, batchsize)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS[1])
end