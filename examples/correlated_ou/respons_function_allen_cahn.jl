using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using HDF5
using ProgressBars

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

# 400 columns is 1 decorrelation time 
experiment_toml = "Experiment_dropout.toml"
FT = Float32

# read experiment parameters from file
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)

batchsize_1 = params.data.batchsize

hfile = h5open("/home/sandre/Repositories/CliMAgen.jl/allen_cahn_timeseries_data.hdf5")
# hfile = h5open("/home/sandre/Repositories/CliMAgen.jl/allen_cahn.hdf5")
X = HDF5.read(hfile, "data")
m, n, ℓ, Ne = size(X)
X = reshape(X, (m,n,1, ℓ*Ne))
Y = copy(X)
exact_score = read(hfile, "score")
close(hfile)

nbatches = floor(Int, size(X)[end]/batchsize_1) # 390

inchannels = params.model.noised_channels
nsamples = params.sampling.nsamples
nimages = params.sampling.nimages
nsteps = params.sampling.nsteps
sampler = params.sampling.sampler
tilesize_sampling = params.sampling.tilesize
dropout_p = params.model.dropout_p

# unpack params
savedir = params.experiment.savedir
rngseed = params.experiment.rngseed
nogpu = params.experiment.nogpu

resolution = params.data.resolution
fraction = params.data.fraction
standard_scaling = params.data.standard_scaling
preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
scaling = JLD2.load_object(preprocess_params_file)

# set up rng
rngseed > 0 && Random.seed!(rngseed)

# set up device
if !nogpu && CUDA.has_cuda()
    device = Flux.gpu
    @info "Sampling on GPU"
else
    device = Flux.cpu
    @info "Sampling on CPU"
end

# let's read in the data here (every dt)

# reshape to 32x32x1xnobs_1
nobs_1 = batchsize_1 * nbatches #size(data)[end]

# carry out preprocessing
X .= apply_preprocessing(X, scaling)
# Allocate memory for the score
data_score = similar(X)
data_score .= FT(0)
# set up model
checkpoint_path = joinpath(savedir, "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth
model = device(model)

# to compute the score
t0 = FT(0.0) # diffusion time, t=0 means data distribution and t=1 means normal distribution
# s(x,t)
for batch in ProgressBar(1:nbatches)
    batch_indices = 1+(batch-1)*batchsize_1:batchsize_1+(batch-1)*batchsize_1
    gpu_data = device(X[:,:,[1],batch_indices])
    data_score[:,:,[1],batch_indices] .= Array(CliMAgen.score(model, gpu_data, t0))
end

#=
@info "saving"
# write new hdf5 with preprocessed data and the score
hfile = h5open("/home/sandre/Repositories/CliMAgen.jl/allen_cahn_generative_kat2.hdf5", "cw")
hfile["timeseries"] = reshape(X, (m,n,ℓ, Ne))
hfile["generative_score"] = reshape(data_score, (m,n, ℓ, Ne))
# hfile["exact_score"] = exact_score * 8
hfile["ensembles"] = Ne
hfile["timesteps"] = ℓ
hfile["last_index"] = batchsize_1+(nbatches-1)*batchsize_1
close(hfile)
@info "done"
=#
##
@info "Calculating correlation"
using LinearAlgebra

#=
X̃ = CuArray(reshape(X, (m * n, ℓ * Ne))[:, 1:ℓ*Ne - 300])
S̃ = CuArray(reshape(data_score, (m * n, ℓ * Ne))[:, 1:ℓ*Ne - 300])
=#

X̃ = CuArray(reshape(Y, (m * n, ℓ * Ne))[:, 1:ℓ*Ne - 300])
S̃ = CuArray(reshape(8/sqrt(2) * exact_score, (m * n, ℓ * Ne))[:, 1:ℓ*Ne - 300])

cor_mat = randn(m * n, m * n)
for i in ProgressBar(1:m*n)
    for j in 1:m*n
        cor_mat[i,j] = -mean(X̃[i,:] .* S̃[j,:])
    end
end
##
identity_error = norm(cor_mat - I) / m*n

