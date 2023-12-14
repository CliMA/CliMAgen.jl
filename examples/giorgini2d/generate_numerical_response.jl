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
include("./trajectory_utils.jl")

FT = Float32

experiment_toml="Experiment.toml"
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

dt = FT(toml_dict["param_group"]["dt"])
dt_save = FT(toml_dict["param_group"]["dt_save"])
T = FT(toml_dict["param_group"]["T"])
# Obtain decorrelation time
fid = h5open(f_path, "r")
decorrelation = read(fid, "decorrelation")
snapshots = read(fid, "snapshots")
close(fid)

M, N, _, L = size(snapshots)

model = setup_ludo_model(σ, α, β, γ, N; FT = Float32)
n_ens = 10^5
ϵ = 0.1
endT = decorrelation*2
tspan = FT.((0.0, endT))
nsteps = Int(floor((tspan[2]-tspan[1])/dt))
n_steps_per_save = Int(round(dt_save/dt))
n_things_in_trajectory = Int(nsteps/n_steps_per_save)
responseN_ens = zeros(N^2, n_things_in_trajectory, n_ens)
for i in ProgressBar(1:n_ens)
    R1 = rand(Int)
    u0 = 2*rand(FT, N^2).-1
    u0 .= atanh(-α/β)/γ
    X0 = simulate(u0, tspan .* 10, dt, dt_save, R1; model = model, progress = false)[:,end]
    X0eps = copy(X0)
    X0eps[1] += ϵ
    random_seed = rand(Int)
    t1 = simulate(X0, tspan, dt, dt_save, random_seed;model=model, progress = false)
    t2 = simulate(X0eps, tspan, dt, dt_save, random_seed; model = model, progress = false)
    responseN_ens[:,:,i] = (t2 .- t1)./ϵ
end
# Means over ensemble members
responseN = mean(responseN_ens, dims=3)[:, :, 1]
err = zeros(N^2, n_things_in_trajectory)
for i in 1:n_things_in_trajectory
    err[:,i] = std(responseN_ens[:,i,:], dims=2)
end

data_directory = joinpath(package_dir, "data")
file_path = joinpath(data_directory, "numerical_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
hfile = h5open(file_path, "w")
hfile["pixel response"] = responseN
hfile["lag_indices"] = collect(1:1:n_things_in_trajectory)
hfile["std_err"] = err ./ sqrt(n_ens)
close(hfile)
