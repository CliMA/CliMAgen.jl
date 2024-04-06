using Statistics
using Random
using ProgressBars
using TOML
using StatsBase
using HDF5
using CliMAgen
using LinearAlgebra

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
M, N, _, L = size(trj)
trj = reshape(trj, (M * N, L))
# decorrelation = read(fid, "decorrelation")
close(fid)

using Random
Random.seed!(1234)
N = size(trj)[2]
list1 = rand(1:N, 1000)
list2 = rand(1:N, 1000)
σmax_guestimate= maximum([norm(trj[:, i] - trj[:, j]) for i in list1, j in list2]) 
println(" σmax_guestimate = $(σmax_guestimate * 1.2)")