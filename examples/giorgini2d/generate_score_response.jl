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

t0 = 0.0
@info "loading score function"
r_trj = reshape(trj, (M, N, 1, L))
x = CuArray(r_trj)
@time scores = Array(CliMAgen.score(model, Float32.(x), t0))
x = copy(trj[:, 1:res:res*t])
sc = reshape(scores, (N * N, t))
xt = transpose(x)
sct = transpose(sc)
responseS = zeros(N^2, N^2, length(lag_indices))

for i in ProgressBar(eachindex(lag_indices))
    li = lag_indices[i]
    responseS[:, :, i] = cov(xt[li:end, :], sct[1:end-li+1, :])
end

responseS_normalized_left = zeros(N^2, N^2, length(lag_indices))
responseS_normalized_right = zeros(N^2, N^2, length(lag_indices))
responseS_normalized_average = zeros(N^2, N^2, length(lag_indices))
for i in ProgressBar(eachindex(lag_indices))
    li = lag_indices[i]
    normalization = inv(responseS[:, :, 1])
    responseS_normalized_left[:, :, i] = responseS[:, :, i] * normalization 
    responseS_normalized_right[:, :, i] = normalization * responseS[:, :, i]
    responseS_normalized_average[:, :, i] = (responseS_normalized_left[:, :, i] + responseS_normalized_right[:, :, i])/2
end

data_directory = joinpath(package_dir, "data")
file_path = joinpath(data_directory, "score_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
hfile = h5open(file_path, "w")
hfile["response"] = responseS
hfile["right normalized response"] = responseS_normalized_right
hfile["left normalized response"] = responseS_normalized_left
hfile["both normalized response"] = responseS_normalized_both
hfile["lag_indices"] = collect(lag_indices)
close(hfile)