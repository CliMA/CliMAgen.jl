using Statistics
using Random
using LinearAlgebra
using ProgressBars
using TOML
using Plots
using StatsBase
using HDF5
using BSON
using Distributions

function set_to_zero_if_smaller(vector, a)
    return [abs(x) < a ? 0 : x for x in vector]
end

function generate_2d_gaussian_lattice(N,mu,sigma)
    x = range(1, stop=N, length=N)
    y = range(1, stop=N, length=N)
    gaussian_distribution = [pdf(MvNormal(mu, sigma), [xi, yi]) for xi in x, yi in y]
    gaussian_distribution ./= sqrt(sum(gaussian_distribution.^2))
    
    return gaussian_distribution
end

package_dir = pwd()
data_directory = joinpath(package_dir, "giorgini2d/data")
model_toml = "giorgini2d/Model.toml"

toml_dict = TOML.parsefile(model_toml)
α = FT(toml_dict["param_group"]["alpha"])
β = FT(toml_dict["param_group"]["beta"])
γ = FT(toml_dict["param_group"]["gamma"])
σ = FT(toml_dict["param_group"]["sigma"])

file_path = joinpath(data_directory, "linear_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
hfile = h5open(file_path, "r")
responseL = read(hfile["response"])
pixelresponseL = read(hfile["pixel response"])
lag_indicesL = read(hfile["lag_indices"])
close(hfile)

file_path = joinpath(data_directory, "numerical_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
hfile = h5open(file_path, "r")
responseN = read(hfile["pixel response"])
lag_indicesN = read(hfile["lag_indices"])
std_err = read(hfile["std_err"])
close(hfile)

file_path = joinpath(data_directory, "score_response_$(α)_$(β)_$(γ)_$(σ).hdf5")
hfile = h5open(file_path, "r")
responseS = read(hfile["response"])
responseS_normalized_right = read(hfile["right normalized response"])
responseS_normalized_left = read(hfile["left normalized response"])
responseS_normalized_average = read(hfile["both normalized response"])
pixelresponseS_normalized_average = read(hfile["pixel response"])
pixelresponseS = read(hfile["pixel response unnormalized"])
lag_indicesS = read(hfile["lag_indices"])
close(hfile)

RL = set_to_zero_if_smaller(pixelresponseL, 0.0)
RN = set_to_zero_if_smaller(responseN, 0.0)
RS = set_to_zero_if_smaller(pixelresponseS_normalized_average, 0.0)
##
ind = 17
plot(lag_indicesL, RL[ind, :], label="Linear")
plot!(lag_indicesN, RN[ind, :], label="Numerical")
plot!(lag_indicesS, RS[ind, :], label="Score")
##
N = 8
mu = [0,0]
sigma = [0.1,0.1]
dx0 = generate_2d_gaussian_lattice(N,mu,sigma)
x = range(1, stop=N, length=N)
y = range(1, stop=N, length=N)
surface(x,y,dx0)
savefig("dx0_sigma$sigma.png")
##

function create_response_smooth(dx0,lags,R)
    N = length(dx0[:,1])
    R_smooth = zeros(N^2, length(lags))
    dx0_1D = reshape(dx0, N^2)
    for t in 1:length(lags)
        for i in 1:N^2
            R_smooth[i, t] = vcat(R[i:end, t],R[1:i-1, t])'*dx0_1D
        end
    end
    return R_smooth
end

R_smoothL = create_response_smooth(dx0,lag_indicesL, pixelresponseL)
R_smoothN = create_response_smooth(dx0,lag_indicesN, responseN)
R_smoothS = create_response_smooth(dx0,lag_indicesS, pixelresponseS_normalized_average)

pl = []
for i in 1:64
    pl_temp = plot(lag_indicesL, R_smoothL[i, :], label="Linear")
    pl_temp = plot!(lag_indicesN, R_smoothN[i, :], label="Numerical")
    pl_temp = plot!(lag_indicesS, R_smoothS[i, :], label="Score")
    push!(pl, pl_temp)
end
plot(pl[1], pl[2], pl[3], pl[4], pl[5], pl[6], pl[7], pl[8], pl[9], pl[10], pl[11], pl[12], pl[13], pl[14], pl[15], pl[16], pl[17], pl[18], pl[19], pl[20], pl[21], pl[22], pl[23], pl[24], pl[25], pl[26], pl[27], pl[28], pl[29], pl[30], pl[31], pl[32], pl[33], pl[34], pl[35], pl[36], pl[37], pl[38], pl[39], pl[40], pl[41], pl[42], pl[43], pl[44], pl[45], pl[46], pl[47], pl[48], pl[49], pl[50], pl[51], pl[52], pl[53], pl[54], pl[55], pl[56], pl[57], pl[58], pl[59], pl[60], pl[61], pl[62], pl[63], pl[64], layout=(8,8), size=(3000,3000),ylims=(0.,1.),linewidth=3)
savefig("R_smooth_sigma$sigma.png")
