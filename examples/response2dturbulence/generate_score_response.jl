using Statistics
using CliMAgen
using CliMAgen: expand_dims, MeanSpatialScaling, StandardScaling, apply_preprocessing, invert_preprocessing
using Random
using ProgressBars
using TOML
using StatsBase
using HDF5
using BSON
using CUDA
using Flux 
using JLD2

include("trajectory_utils.jl")
# run from response2dturbulence
package_dir = pwd()
FT = Float32
experiment_toml="Experiment.toml"
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
model_savedir = params.experiment.savedir
preprocess_params_file = joinpath(model_savedir, "preprocessing_standard_scaling_false.jld2")
scaling = JLD2.load_object(preprocess_params_file)
savedir = pwd() * "/data/" 
f_path = savedir  * "two_dimensional_turbulence_with_condensation.hdf5"

# Obtain precomputed trajectory, reshape
fid = h5open(f_path, "r")
trj = read(fid, keys(fid)[1])
trj = trj[1:end, 1:end, :, 1000:end]
trj = apply_preprocessing(trj, scaling)
M, N, S, L = size(trj)
r_trj = reshape(trj, (M*N*S, L))
close(fid)
lag_indices = 1:40

# read in model
checkpointdir = joinpath(pwd(),model_savedir)
checkpoint_path = joinpath(checkpointdir , "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth
# pass model to GPU
dev = Flux.gpu
model = dev(model)


t0 = FT(0.0)
@info "loading score function"
decorrelation = 1
decorrelation = max(floor(Int, decorrelation/10), 1)
decorrelated_indices = 1:decorrelation:L
r_trj_decorrelated = r_trj[:, :, :, decorrelated_indices]
L_decorrelated = size(r_trj_decorrelated )[end]
batchsize = 128 
nbatches = floor(Int, L_decorrelated / batchsize)
# t = dev(fill!(similar(x, batchsize), FT(0)))
# need to apply in batches
scores = zeros(Float32, M, N, S, nbatches * batchsize)
for i in ProgressBar(1:nbatches)
    x = dev(CuArray(r_trj_decorrelated[:, :, :, (i-1)*batchsize+1:i*batchsize]))
    scores[:, :, :, (i-1)*batchsize+1:i*batchsize] = Array(CliMAgen.score(model, x, t0))
end

sct = transpose(reshape(scores, (M*N*S, nbatches * batchsize)))
responseS = zeros(M*N*S, M*N*S, length(lag_indices))

for i in ProgressBar(eachindex(lag_indices))
    li = lag_indices[i]
    xt = transpose(r_trj[:, li:decorrelation:end])
    L = min(nbatches*batchsize, size(xt)[1])
    responseS[:, :, i] = -cov(xt[1:L, :], sct[1:decorrelation:L, :])
end


responseS_normalized_left = zeros(M * N * S, M * N * S, length(lag_indices))
responseS_normalized_right = zeros(M * N * S, M * N * S, length(lag_indices))
responseS_normalized_average = zeros(M * N * S, M * N * S, length(lag_indices))
for i in ProgressBar(eachindex(lag_indices))
    normalization = inv(responseS[:, :, 1])
    responseS_normalized_left[:, :, i] = responseS[:, :, i] * normalization 
    responseS_normalized_right[:, :, i] = normalization * responseS[:, :, i]
    responseS_normalized_average[:, :, i] = (responseS_normalized_left[:, :, i] + responseS_normalized_right[:, :, i])/2
end
