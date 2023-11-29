using Statistics
using Random
using LinearAlgebra
using ProgressBars
using TOML
using Plots
using StatsBase
using HDF5
using BSON
using CliMAgen
using JLD2
using CliMAgen: expand_dims, MeanSpatialScaling, StandardScaling, apply_preprocessing, invert_preprocessing
using Distributions
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include("./utils.jl")
include("./rhs.jl")

FT = Float32

toml_dict = TOML.parsefile("giorgini2d/Model.toml")
α = FT(toml_dict["param_group"]["alpha"])
β = FT(toml_dict["param_group"]["beta"])
γ = FT(toml_dict["param_group"]["gamma"])
σ = FT(toml_dict["param_group"]["sigma"])
T = toml_dict["param_group"]["T"]
dt = FT(toml_dict["param_group"]["dt"])
dt_save = FT(toml_dict["param_group"]["dt_save"])
N = toml_dict["param_group"]["N"]
seed = toml_dict["param_group"]["seed"]

pfile = JLD2.load_object("giorgini2d/preprocessing_standard_scaling_false.jld2")

function scaling2D(x; pfile=pfile, FT=Float32)
    return reshape(apply_preprocessing(reshape(x,(N,N,1,size(x)[2])), pfile), (N^2,size(x)[2]))
end

Γ = FT.(reshape(zeros(N^4), (N^2,N^2)))
for i1 in 1:N
    for j1 in 1:N
        k1 = (j1-1)*N+i1
        for i2 in 1:N
            for j2 in 1:N
                k2 = (j2-1)*N+i2
                Γ[k1,k2] = FT(1/sqrt(min(abs(i1-i2),N-abs(i1-i2))^2 + min(abs(j1-j2),N-abs(j1-j2))^2+1))
            end
        end
    end
end
ΓL = cholesky(Γ).L
W = zeros(FT, N*N)
W_corr = similar(W)

tspan = FT.((0.0,T))
u = 2*rand(FT, N^2).-1
model = LudoDiffusionSDE(σ, α, β, γ, N, Periodic(), ΓL, W, W_corr)
Random.seed!(seed)
deterministic_tendency! = make_deterministic_tendency(model)
stochastic_increment! = make_stochastic_increment(model)
du = similar(u)
nsteps = Int(floor((tspan[2]-tspan[1])/dt))
n_steps_per_save = Int(round(dt_save/dt))
savesteps = 0:n_steps_per_save:nsteps - n_steps_per_save
solution = zeros(FT, (N^2, Int(nsteps/n_steps_per_save)))
solution[:, 1] .= reshape(u, (N^2,))
for i in 1:nsteps
    t = tspan[1]+dt*(i-1)
    Euler_Maruyama_step!(du, u, t, deterministic_tendency!, stochastic_increment!, dt)
    if i ∈ savesteps
        save_index = Int(i/n_steps_per_save)
        solution[:, save_index+1] .= reshape(u, (N^2,))
    end
end

solution = scaling2D(solution)

file_path = "data/data_$(α)_$(β)_$(γ)_$(σ).hdf5"
hfile = h5open(file_path, "w") 
hfile["timeseries"] = reshape(solution,(N,N,size(solution)[2]))
close(hfile)