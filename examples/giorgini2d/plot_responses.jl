# This code computes the response functions numerically, using the linear approximation and the score function
using TOML
using CairoMakie
using HDF5

# run from giorgini2d
package_dir = pwd()

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
data_directory = joinpath(package_dir, "data")
numerical_response_path = joinpath(data_directory, "numerical_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
linear_response_path = joinpath(data_directory, "linear_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
score_response_path = joinpath(savedir, "score_response_$(α)_$(β)_$(γ)_$(σ).hdf5")

fid = h5open(numerical_response_path, "r")
responseN = read(fid, "response")
lagsN = read(fid, "lag_indices")
std_err = read(fid, "std_err")
close(fid)

fid = h5open(linear_response_path, "r")
responseL = read(fid, "response")
lagsL = read(fid, "lag_indices")
close(fid)

fid = h5open(score_response_path, "r")
responseS = read(fid, "response")
lagsS = read(fid, "lag_indices")
close(fid)
fig = Figure(resolution=(400, 400), fontsize=24)
i = 1
ax = Axis(fig[1,1], xlabel="Lag", ylabel="Response", title="1-$(i)", titlefont = :regular)
band!(lagsN, responseN[i,:].-std_err[i,:], responseN[i,:].+std_err[i,:], color=(:orange, 0.3), label="Numerical")
lines!(lagsN, responseN[i,:].-std_err[i,:], color=(:orange, 0.5), strokewidth = 1.5)
lines!(lagsN, responseN[i,:].+std_err[i,:], color=(:orange, 0.5), strokewidth = 1.5)
lines!(lagsS, responseS[i,:], color=(:purple, 0.5), strokewidth = 1.5, label = "Score Model")
#check this indexing
lines!(lagsL, responseL[i,i,:], color=(:green, 0.5), strokewidth = 1.5, label = "Linear")
axislegend(; position= :lt, labelsize=16)

save("comparison.png", fig, px_per_unit = 2)