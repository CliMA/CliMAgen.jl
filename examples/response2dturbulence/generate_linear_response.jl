using Statistics
using Random
using ProgressBars
using TOML
using StatsBase
using HDF5
using CliMAgen


package_dir = pwd()
FT = Float32

savedir = pwd() * "/data/" 
f_path = savedir  * "two_dimensional_turbulence_with_condensation.hdf5"

# Obtain precomputed trajectory, reshape
fid = h5open(f_path, "r")
trj = read(fid, keys(fid)[1])
stride = 1
trj = trj[1:stride:end, 1:stride:end, :, 1000:end]
M, N, S, L = size(trj)
trj = Float64.(reshape(trj, (M * N * S, L)))
close(fid)
lag_indices = 1:40

# Linear response estimate
#=
# correct way for the future?
U, Σ, V = svd(trj)
retained_modes = sum(Σ / Σ[1] .> sqrt(eps(1000.0)))
modes = V[:, 1:retained_modes]
=#
trj_t = transpose(trj) # .- mean(trj)
li = 1
C0 = cov(trj_t[li:end, :], trj_t[1:end-li+1, :]) + I * 1e-6
invC0 = inv(C0)
responseL = zeros(Float64, M * N * S, M * N * S, length(lag_indices))
# For all lags, compute response function for all pixels and for all non-overlapping
# segments of length tau in the timeseries
for i in ProgressBar(eachindex(lag_indices)) # ProgressBar(eachindex(lag_indices))
    li = lag_indices[i]
    responseL[:, :, i] = (cov(trj_t[li:end, :], trj_t[1:end-li+1, :])  + I * 1e-6)* invC0 
end

file_path = joinpath(savedir, "linear_response_2Dturbulence.hdf5")
hfile = h5open(file_path, "w")
hfile["response"] = responseL
hfile["lag_indices"] = collect(lag_indices)
close(hfile)