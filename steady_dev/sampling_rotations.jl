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
import CliMAgen: GaussianFourierProjection

include("sampler.jl")


device = Flux.gpu

function load_model(checkpoint_path)
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    score_model_smooth = device(model_smooth)
    return score_model_smooth
end

model = "checkpoint_conditional_rotations_steady_online_timestep_180000.bson"
model = "checkpoint_conditional_rotations_steady_online_timestep_400000.bson"
model = "checkpoint_conditional_rotations_steady_online_timestep_960000.bson"
model = "checkpoint_capacity_conditional_rotations_steady_online_timestep_700000.bson"

function GaussianFourierProjection(embed_dim::Int, embed_dim2::Int, scale::FT) where {FT}
    Random.seed!(1234) # same thing every time
    W = randn(FT, embed_dim ÷ 2, embed_dim2) .* scale
    return GaussianFourierProjection(W)
end

gfp = GaussianFourierProjection(128, 64, 30.0f0)
@info "loading model"
score_model_smooth_s = load_model(model)

@info "setting up sampler"
nsamples = 100
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
rotation_rates = [Float32(0.6e-4), Float32(1.1e-4), Float32(1.5e-4), Float32(7.29e-5)]
a = 5e-5 
b = 1e-4
times = (rotation_rates .- a) ./ (b-a)
sample_list = []
for i in eachindex(rotation_rates)
    ĉ = reshape(gfp(times[i]), 128, 64, 1, 1)
    c = zeros(Float32, 128, 64, 1, nsamples)
    c .= ĉ
    @info "sampling $(rotation_rates[i])"
    samples = Array(Euler_Maruyama_sampler(score_model_smooth_s, init_x, time_steps, Δt; rng, c))
    hfile = h5open("rotation_rate_samples_$i.hdf5", "w")
    hfile["samples"] = samples
    hfile["rotation"] = rotation_rates[i]
    close(hfile)
    push!(sample_list, copy(samples))
end

##
fig = Figure() 
ax = Axis(fig[1,1])
hist!(ax, sample_list[1][:]; normalization = :pdf, bins = 100, color= (:red, 0.5))
hist!(ax, sample_list[2][:]; normalization = :pdf, bins = 100, color= (:green, 0.5))
hist!(ax, sample_list[3][:]; normalization = :pdf, bins = 100, color= (:gray, 0.5))
hist!(ax, sample_list[end][:]; normalization = :pdf, bins = 100, color= (:blue, 0.5))
ax = Axis(fig[1,2])
hist!(ax, sample_list[1][:]; normalization = :pdf, bins = 100, color= (:red, 0.5))
ax = Axis(fig[1,3])
hist!(ax, sample_list[2][:]; normalization = :pdf, bins = 100, color= (:green, 0.5))
ax = Axis(fig[1,4])
hist!(ax, sample_list[3][:]; normalization = :pdf, bins = 100, color= (:gray, 0.5))
ax = Axis(fig[1,5])
hist!(ax, sample_list[end][:]; normalization = :pdf, bins = 100, color= (:blue, 0.5))
display(fig)
save("rotation_samples.png", fig)

##
hfile = h5open("rotation_rate_samples_1.hdf5", "r")
timeseries1 = read(hfile["samples"])
close(hfile)
hfile = h5open("rotation_rate_data_2.hdf5", "r")
timeseries2 = read(hfile["timeseries"])
close(hfile)
hfile = h5open("rotation_rate_data_3.hdf5", "r")
timeseries3 = read(hfile["timeseries"])
close(hfile)
hfile = h5open("rotation_rate_data_4.hdf5", "r")
timeseries4 = read(hfile["timeseries"])
close(hfile)

fig = Figure() 
ax = Axis(fig[1,1])
hist!(ax, sample_list[1][:]; normalization = :pdf, bins = 100, color= (:red, 0.5))
hist!(ax, sample_list[2][:]; normalization = :pdf, bins = 100, color= (:green, 0.5))
hist!(ax, sample_list[3][:]; normalization = :pdf, bins = 100, color= (:gray, 0.5))
hist!(ax, sample_list[end][:]; normalization = :pdf, bins = 100, color= (:blue, 0.5))
ax = Axis(fig[1,2])
hist!(ax, sample_list[1][:]; normalization = :pdf, bins = 100, color= (:red, 0.5))
ax = Axis(fig[1,3])
hist!(ax, sample_list[2][:]; normalization = :pdf, bins = 100, color= (:green, 0.5))
ax = Axis(fig[1,4])
hist!(ax, sample_list[3][:]; normalization = :pdf, bins = 100, color= (:gray, 0.5))
ax = Axis(fig[1,5])
hist!(ax, sample_list[end][:]; normalization = :pdf, bins = 100, color= (:blue, 0.5))
display(fig)
ax = Axis(fig[2,1])
hist!(ax, timeseries1[:]; normalization = :pdf, bins = 100, color= (:red, 0.5))
hist!(ax, timeseries2[:]; normalization = :pdf, bins = 100, color= (:green, 0.5))
hist!(ax, timeseries3[:]; normalization = :pdf, bins = 100, color= (:gray, 0.5))
hist!(ax, timeseries4[:]; normalization = :pdf, bins = 100, color= (:blue, 0.5))
display(fig)
ax = Axis(fig[2, 2])
hist!(ax, timeseries1[:]; normalization = :pdf, bins = 100, color= (:red, 0.5))
ax = Axis(fig[2, 3])
hist!(ax, timeseries2[:]; normalization = :pdf, bins = 100, color= (:green, 0.5))
ax = Axis(fig[2, 4])
hist!(ax, timeseries3[:]; normalization = :pdf, bins = 100, color= (:gray, 0.5))
ax = Axis(fig[2, 5])
hist!(ax, timeseries4[:]; normalization = :pdf, bins = 100, color= (:blue, 0.5))
save("rotation_samples_comparison.png", fig)
