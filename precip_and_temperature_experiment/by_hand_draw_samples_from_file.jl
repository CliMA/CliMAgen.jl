using CliMAgen, Flux, HDF5, Random, ProgressBars, LinearAlgebra, Statistics, BSON

const gfp_scale = 1
Random.seed!(1234)
const extra_scale = 1

# train differently, t = 0 and t = 1 
# condition using different information (such as global and ensemble average mean surface)

include("process_data.jl")
# load data
FT = Float32
hfile = h5open("/nobackup1/users/sandre/GaussianEarth/pr_field_month_1.hdf5", "r")
physical_sigma = FT.(read(hfile["std"]) * extra_scale) 
physical_mu = FT.(read(hfile["mean"]))
oldfield = FT.(read(hfile["timeseries"]) / extra_scale) 
sigma_max =  FT(read(hfile["max distance"] )  / extra_scale)
tas_rescaled = read(hfile["tasrescaled"])
close(hfile)

inds = vcat(collect(1:30), collect(222:251))
inds = 1:251
N = length(inds)
oldfield2 = reshape(reshape(oldfield, (192, 96, 251, 45))[:, :, inds, :], 192, 96, 1, length(inds) * 45)
field = gmt_embedding_4(oldfield2, tas_rescaled, gfp; N) # gmt_embedding(oldfield, tas_rescaled, gfp)
ensemble_mean = reshape(mean(reshape(oldfield, 192, 96, 251, 45), dims = 4), (192, 96, 1, 251))
contextfield = reshape(mean(reshape(field[:,:, 2, : ], 192, 96, 251, 45), dims = 4), (192, 96, 1, 251))

##
device = Flux.gpu
nwarmup = 5000
gradnorm = FT(1.0);
learning_rate = FT(2e-4);
beta_1 = FT(0.9);
beta_2 = FT(0.999);
epsilon = FT(1e-8);
ema_rate = FT(0.999);
# Optimization
device = Flux.gpu
inchannels = 1
context_channels = 1
sigma_min = FT.(1e-2)
sigma_max = FT.(sigma_max)
##
checkpoint_path = "pr_500.bson"
checkpoint_path = "experiment2_" * checkpoint_path
BSON.@load checkpoint_path model model_smooth opt opt_smooth
score_model_smooth = device(model_smooth)
##
include("byhand_draw_samples.jl")