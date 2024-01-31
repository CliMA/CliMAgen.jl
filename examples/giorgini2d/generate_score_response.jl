using Statistics
using Random
using ProgressBars
using TOML
using Plots
using StatsBase
using HDF5
using BSON
using CliMAgen
using CliMAgen: expand_dims, MeanSpatialScaling, StandardScaling, apply_preprocessing, invert_preprocessing
using Distributions
using CUDA
using Flux 

# run from giorgini2d
package_dir = pwd()
include("./trajectory_utils.jl")

FT = Float32

experiment_toml = "Experiment.toml"
model_toml = "Model.toml"

toml_dict = TOML.parsefile(model_toml)
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
α = FT(toml_dict["param_group"]["alpha"])
β = FT(toml_dict["param_group"]["beta"])
γ = FT(toml_dict["param_group"]["gamma"])
σ = FT(toml_dict["param_group"]["sigma"])
savedir = "$(params.experiment.savedir)_$(α)_$(β)_$(γ)_$(σ)"
f_path = "data/data_$(α)_$(β)_$(γ)_$(σ).hdf5"

fid = h5open(f_path, "r")
trj = read(fid, "timeseries")
M, N, L = size(trj)
trj = reshape(trj, (M * N, L))
decorrelation = read(fid, "decorrelation")
close(fid)
lag_indices = 1:4:decorrelation*2
# Compute using the score function
checkpoint_path = joinpath(savedir, "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth
dev = Flux.gpu
model = dev(model)
t0 = 0.0
@info "loading score function"
r_trj = reshape(trj, (M, N, 1, L))
decorrelation = max(floor(Int, decorrelation/10), 1)
decorrelated_indices = 1:decorrelation:L
r_trj_decorrelated = r_trj[:, :, [1], decorrelated_indices]
L_decorrelated = size(r_trj_decorrelated )[end]
batchsize = 32
nbatches = floor(Int, L_decorrelated / batchsize)
# t = dev(fill!(similar(x, batchsize), FT(0)))
# need to apply in batches
scores = zeros(Float32, M, N, 1, nbatches * batchsize)
for i in ProgressBar(1:nbatches)
    x = dev(CuArray(r_trj_decorrelated[:, :, [1], (i-1)*batchsize+1:i*batchsize]))
    scores[:, :, 1, (i-1)*batchsize+1:i*batchsize] = Array(CliMAgen.score(model, x, Float32(0.0)))
end

sct = transpose(reshape(scores, (M*N, nbatches * batchsize)))
responseS = zeros(M*N, M*N, length(lag_indices))

for i in ProgressBar(eachindex(lag_indices))
    li = lag_indices[i]
    xt = transpose(trj[:, li:decorrelation:end])
    L = min(nbatches*batchsize, size(xt)[1])
    responseS[:, :, i] = -cov(xt[1:L, :], sct[1:L, :])
end

responseS_normalized_left = zeros(N^2, N^2, length(lag_indices))
responseS_normalized_right = zeros(N^2, N^2, length(lag_indices))
responseS_normalized_average = zeros(N^2, N^2, length(lag_indices))
for i in ProgressBar(eachindex(lag_indices))
    normalization = inv(responseS[:, :, 1])
    responseS_normalized_left[:, :, i] = responseS[:, :, i] * normalization 
    responseS_normalized_right[:, :, i] = normalization * responseS[:, :, i]
    responseS_normalized_average[:, :, i] = (responseS_normalized_left[:, :, i] + responseS_normalized_right[:, :, i])/2
end

pixelresponseS_normalized_average = responseS_normalized_average[:, 1, :]
pixelresponseS = responseS[:, 1, :]

savedir = "$(params.experiment.savedir)_$(α)_$(β)_$(γ)_$(σ)"
data_directory = joinpath(package_dir, savedir)
file_path = joinpath(data_directory, "score_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
hfile = h5open(file_path, "w")
hfile["response"] = responseS
hfile["right normalized response"] = responseS_normalized_right
hfile["left normalized response"] = responseS_normalized_left
hfile["both normalized response"] = responseS_normalized_average
hfile["pixel response"] = pixelresponseS_normalized_average
hfile["pixel response unnormalized"] = pixelresponseS
hfile["lag_indices"] = collect(lag_indices)
close(hfile)
