using LinearAlgebra
using CairoMakie
using Statistics
using ProgressBars
using Flux
using CliMAgen
using BSON
using HDF5

include("sampler.jl")

device = Flux.gpu

function load_model(checkpoint_path)
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    score_model_smooth = device(model_smooth)
    return score_model_smooth
end

score_model_smooth = load_model("steady_state_fixed_data_epoch_100.bson")

nsamples = 16
nsteps = 250
inchannels = 1
Ny = 64
resolution = (128, 64)
time_steps, Δt, init_x = setup_sampler(
    score_model_smooth,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)

samples = Euler_Maruyama_sampler(score_model_smooth, init_x, time_steps, Δt)

hfile = h5open("steady_default_data.hdf5", "r")
timeseries = read(hfile["timeseries"])
μ = read(hfile, "shift")
σ = read(hfile, "scaling")
lon = read(hfile["lon"])
lat = read(hfile["lat"])
close(hfile)

physical_timeseries = σ .* timeseries .+ μ
physical_samples = Array(σ .* samples .+ μ)

##
ensemble_indices = collect(1:nsamples)
M = floor(Int, sqrt(length(ensemble_indices)))

fig = Figure(resolution = (800, 800))
for (i,j) in enumerate(ensemble_indices)
    ii = (i-1)÷M  + 1
    jj = (i-1)%M  + 1
    ax = Axis(fig[ii, jj]; xlabel = "Longitude", ylabel = "Latitude")
    heatmap!(ax, lon, lat, physical_samples[:,:,1,j], colormap = :thermometer, colorrange = (215, 291))
end

save("temperature_heatmap_samples.png", fig)