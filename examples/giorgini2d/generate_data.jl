using Random, Statistics, StatsBase
using ProgressBars
using TOML
using HDF5
using JLD2
using CliMAgen: MeanSpatialScaling, invert_preprocessing, apply_preprocessing
import CliMAgen
package_dir = pwd() # pkgdir(CliMAgen)
include("./trajectory_utils.jl")

function preprocess_data!(timeseries; preprocess_params_file="preprocess_params.jld2")
    N², M = size(timeseries)
    N = round(Int, sqrt(N²))
    xtrain = reshape(timeseries, (N, N, 1, M))
    #scale means and spatial variations separately
    x̄ = mean(xtrain, dims=(1, 2))
    maxtrain_mean = maximum(x̄, dims=4)
    mintrain_mean = minimum(x̄, dims=4)
    Δ̄ = maxtrain_mean .- mintrain_mean
    xp = xtrain .- x̄
    maxtrain_p = maximum(xp, dims=(1, 2, 4))
    mintrain_p = minimum(xp, dims=(1, 2, 4))
    Δp = maxtrain_p .- mintrain_p

    # To prevent dividing by zero
    Δ̄[Δ̄ .== 0] .= FT(1)
    Δp[Δp .== 0] .= FT(1)
    scaling = MeanSpatialScaling{FT}(mintrain_mean, Δ̄, mintrain_p, Δp)
    JLD2.save_object(preprocess_params_file, scaling)
    xtrain .= apply_preprocessing(xtrain, scaling)
    timeseries .= reshape(xtrain, (N², M))
    return nothing
end

# run in folder giorgini2d
# Set up parameters and float type
FT = Float32
toml_dict = TOML.parsefile("Model.toml")
α = FT(toml_dict["param_group"]["alpha"])
β = FT(toml_dict["param_group"]["beta"])
γ = FT(toml_dict["param_group"]["gamma"])
σ = FT(toml_dict["param_group"]["sigma"])
T = FT(toml_dict["param_group"]["T"]) # Total time
dt = FT(toml_dict["param_group"]["dt"]) # timestep 
dt_save = FT(toml_dict["param_group"]["dt_save"]) # timestep between saves
N = toml_dict["param_group"]["N"] # size of image in one direction
seed = toml_dict["param_group"]["seed"]

# Setup simulation
model = setup_ludo_model(σ, α, β, γ, N; FT = Float32)

# Set tspan and initial condition
tspan = FT.((0.0,T))
u = 2*rand(FT, N^2).-1
u .= atanh(-α/β)/γ

# Run simulation
solution = simulate(u, tspan, dt, dt_save, seed; model)

# Compute autocorrelation
lags = [0:2000...]
ac = StatsBase.autocor(solution[1,1:length(lags)*100], lags; demean = true)
τ = minimum(lags[ac .< 0.1]) # in units of dt_save
# Our initial condition should minimize spinup, but to be safe, remove the first 10 autocorrelation times
solution = solution[:, 10*τ:end]

# add thing to make directory if it doesn't exist
data_directory = joinpath(package_dir, "data")
if isdir(data_directory) == false
    mkdir(data_directory)
end
preprocess_params_file = joinpath(data_directory, "preprocess_params.jld2")

# compute preprocessing parameters and preproccess the data
preprocess_data!(solution; preprocess_params_file = preprocess_params_file)
decorrelated_solution = solution[:, 1:τ:end]
M = size(decorrelated_solution)[end]
# σmax = maximum([norm(decorrelated_solution[:, i] - decorrelated_solution[:,j]) for i in rand(1:M, 5000), j in rand(1:M, 5000)])
file_path = joinpath(data_directory,"data_$(α)_$(β)_$(γ)_$(σ).hdf5")
hfile = h5open(file_path, "w") 
hfile["timeseries"] = reshape(solution,(N,N,size(solution)[2]))
hfile["snapshots"] = reshape(decorrelated_solution,(N,N,1,size(decorrelated_solution)[2]))
hfile["decorrelation"] = τ
hfile["decorrelation time"] = τ * dt_save
hfile["dt save"] = dt_save
close(hfile)