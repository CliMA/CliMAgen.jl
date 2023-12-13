# This code computes the response functions numerically, using the linear approximation and the score function
using Statistics
using Random
using ProgressBars
using TOML
using StatsBase
using HDF5
using CliMAgen

# run from giorgini2d
package_dir = pwd()

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

# Obtain precomputed trajectory, reshape
fid = h5open(f_path, "r")
trj = read(fid, "timeseries")
M, N, L = size(trj)
trj = reshape(trj, (M * N, L))
decorrelation = read(fid, "decorrelation")
close(fid)
lag_indices = 1:4:decorrelation*2

# Linear response estimate
trj_t = transpose(trj) # .- mean(trj)
invC0 = inv(cov(trj_t))
responseL = zeros(N^2, N^2, length(lag_indices))
# For all lags, compute response function for all pixels and for all non-overlapping
# segments of length tau in the timeseries
for i in ProgressBar(eachindex(lag_indices))
    li = lag_indices[i]
    responseL[:, :, i] = cov(trj_t[li:end, :], trj_t[1:end-li+1, :]) * invC0
end

data_directory = joinpath(package_dir, "data")
file_path = joinpath(data_directory, "linear_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
hfile = h5open(file_path, "w")
hfile["response"] = responseL
hfile["pixel response"] = responseL[1, :, :]
hfile["lag_indices"] = collect(lag_indices)
close(hfile)