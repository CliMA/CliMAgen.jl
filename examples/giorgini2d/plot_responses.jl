# This code computes the response functions numerically, using the linear approximation and the score function
using TOML
using GLMakie
using HDF5
using CliMAgen

# run from giorgini2d
package_dir = pwd()

experiment_toml="Experiment.toml"
model_toml = "Model.toml"
FT = Float32
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

@info "loading numerical response"
fid = h5open(numerical_response_path, "r")
responseN = read(fid, "pixel response")
lagsN = read(fid, "lag_indices")
std_err = read(fid, "std_err")
close(fid)

@info "loading linear response"
fid = h5open(linear_response_path, "r")
responseL = read(fid, "pixel response")
lagsL = read(fid, "lag_indices")
close(fid)

@info "loading score response"
fid = h5open(score_response_path, "r")
responseS = read(fid, "pixel response")
responseS2 = read(fid, "pixel response unnormalized")
lagsS = read(fid, "lag_indices")
close(fid)
fig = Figure(resolution=(1600, 1600), fontsize=24)
N = 8
for i in 1:N^2
    ii = floor(Int, (i-1)/N) + 1
    jj = mod(i-1, N) + 1
    ax = Axis(fig[ii,jj], xlabel="Lag", ylabel="Response", title="1-$(i)", titlefont = :regular)
    band!(lagsN, responseN[i,:].-std_err[i,:], responseN[i,:].+std_err[i,:], color=(:orange, 0.3), label="Numerical")
    lines!(lagsN, responseN[i,:].-std_err[i,:], color=(:orange, 0.5), strokewidth = 1.5)
    lines!(lagsN, responseN[i,:].+std_err[i,:], color=(:orange, 0.5), strokewidth = 1.5)
    lines!(lagsS, responseS[i,:], color=(:purple, 0.5), strokewidth = 1.5, label = "Score Model (hack)")
    # scatter!(lagsS, responseS2[i,:], color=(:green, 0.1), strokewidth = 1.5, label = "Score Model (no hack)")
   # GLMakie.ylims!(ax, (-1.0, 1.0))
    #check this indexing
    lines!(lagsL, responseL[i, :], color=(:green, 0.5), strokewidth = 1.5, label = "Linear")
    if i == 64
        axislegend(; position= :rt, labelsize=16)
        hidedecorations!(ax)
    else
        hidedecorations!(ax)
    end
end
display(fig)
save("comparison.png", fig, px_per_unit = 2)