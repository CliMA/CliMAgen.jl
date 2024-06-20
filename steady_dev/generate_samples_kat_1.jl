using ProgressBars
using Flux
using CliMAgen
using BSON
using HDF5
using CUDA
using Random

best_epoch = 100
nsamples = 10000
nbatches = 40
batchsize = 250
nsteps = 250
inchannels = 1
resolution = (128, 64)
FT = Float64
samples = zeros(FT, (resolution...,inchannels, nsamples))

device = Flux.gpu
include("../sandbox/sampler.jl")
function load_model(checkpoint_path)
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    score_model_smooth = device(model_smooth)
    return score_model_smooth
end

data_directory = ""

if fixed_model
    @info "Loading fixed model"
    model = data_directory * "steady_state_fixed_data_epoch_$best_epoch.bson"
    sample_hdf5 = data_directory * "steady_state_fixed_data_epoch_$(best_epoch)_samples.hdf5"
    seed = 17
else
    if capacity
        @info "Loading online model - high capacity"
        model = data_directory * "checkpoint_capacity_steady_online_timestep_400000.bson"
        sample_hdf5 = data_directory * "checkpoint_capacity_steady_online_timestep_400000_samples.hdf5"
        seed = 19
    else
        @info "Loading online model - low capacity"
        model = data_directory * "checkpoint_steady_online_timestep_400000.bson"
        sample_hdf5 = data_directory * "checkpoint_steady_online_timestep_400000_samples.hdf5"
        seed = 23
    end
end


score_model_smooth = load_model(model)
for batch in 1:nbatches
    @info batch
    time_steps, Δt, init_x = setup_sampler(
        score_model_smooth,
        device,
        resolution,
        inchannels;
        num_images=batchsize,
        num_steps=nsteps,
    )
    rng = MersenneTwister(batch*seed)
    samples[:,:,:,(batch-1)*batchsize+1:batch*batchsize] = Array(Euler_Maruyama_sampler(score_model_smooth, init_x, time_steps, Δt; rng))
end
fid = HDF5.h5open(sample_hdf5, "w")
fid["generated_raw_samples"] = samples
close(fid)