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
experiment_toml = "Experiment.toml"
FT = Float32

# read experiment parameters from file
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)

batchsize_1 = params.data.batchsize

hfile = h5open("/home/sandre/Repositories/CliMAgen.jl/x13data.hdf5")
X = HDF5.read(hfile, "x")
# close(hfile)

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
# preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
# scaling = JLD2.load_object(preprocess_params_file)

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

# The below is usually carried out by the dataloader; shhhhh
# data_directory = "/home/sandre/.julia/datadeps/CorrelatedOU2D/grf_dt_save_eq_dt.hdf5"
# hfile = h5open(data_directory, "r")
# data = read(hfile["res_32x32_1.0_0.5_0.25_0.25"]);

# reshape to 32x32x1xnobs_1
nobs_1 = batchsize_1 * nbatches #size(data)[end]
X = reshape(X[:,1:nobs_1], (8,8,1,nobs_1))

# carry out preprocessing
# data .= apply_preprocessing(data, scaling)
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

@info "saving"
# write new hdf5 with preprocessed data and the score
hfile = h5open("/home/sandre/Repositories/CliMAgen.jl/nonlineardata.hdf5", "cw")
hfile["timeseries"] = X
hfile["score"] = data_score 
close(hfile)
@info "done"
