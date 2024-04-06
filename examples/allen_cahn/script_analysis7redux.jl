using BSON
using Flux
using CUDA
using cuDNN
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using CliMAgen
using ProgressBars

# run from giorgni2d
include("../utils_data.jl") # for data loading
include("../utils_analysis.jl") # for data loading
include("dataloader.jl") # for data loading

model_toml = "Model7.toml"
experiment_toml = "Experiment7.toml"
FT = Float32
toml_dict = TOML.parsefile(model_toml)
α = FT(toml_dict["param_group"]["alpha"])
β = FT(toml_dict["param_group"]["beta"])
γ = FT(toml_dict["param_group"]["gamma"])
σ = FT(toml_dict["param_group"]["sigma"])
f_path = "data/data_$(α)_$(β)_$(γ)_$(σ)_context.hdf5"

# read experiment parameters from file
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)

savedir = "$(params.experiment.savedir)_$(α)_$(β)_$(γ)_$(σ)"
# set up directory for saving checkpoints
!ispath(savedir) && mkpath(savedir)


rngseed = params.experiment.rngseed
nogpu = params.experiment.nogpu

nsamples = params.data.batchsize
fraction = params.data.fraction

inchannels = params.model.noised_channels
noised_channels = inchannels
context_channels = params.model.context_channels
nsamples = params.sampling.nsamples
nimages = params.sampling.nimages
nsteps = params.sampling.nsteps
sampler = params.sampling.sampler
rngseed > 0 && Random.seed!(rngseed)

# set up device
if !nogpu && CUDA.has_cuda()
    device = Flux.gpu
    @info "Sampling on GPU"
else
    device = Flux.cpu
    @info "Sampling on CPU"
end


checkpoint_path = joinpath(savedir, "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth
model = device(model_smooth)

# sample from the trained model
nsamples = 30
resolution = 96
time_steps, Δt, init_x = setup_sampler(
    model,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)

hfile = h5open("data/tmpcheck.hdf5", "r");
snapshots = read(hfile["snapshots"]);
μ = read(hfile["mean"])
scale = read(hfile["scale"])
close(hfile)


unique_context_index = collect(1:30:2580)
data_context = copy(snapshots[:, :, (noised_channels+1):(noised_channels+context_channels), 1:nsamples])
generated_samples = copy(snapshots) * 0 

rng = MersenneTwister(1234)
for i in ProgressBar(eachindex(unique_context_index))
    data_context .= snapshots[:, :, (noised_channels+1):(noised_channels+context_channels), [unique_context_index[i]]]
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=data_context[:, :, :, 1:nsamples], rng=rng)
    samples = cpu(samples)
    @. generated_samples[:, :, 1:noised_channels, 1 + (i-1) * nsamples:nsamples * i] = samples
    @. generated_samples[:, :, (noised_channels+1):(noised_channels+context_channels), 1+(i-1)*nsamples:nsamples*i] = data_context
    generated_samples[:, :, :, 1 + (i-1) * nsamples:nsamples * i] .*= scale 
    generated_samples[:, :, :, 1 + (i-1) * nsamples:nsamples * i] .+= μ
end
hfile = h5open("generated_samples.hdf5", "w")
hfile["snapshots"] = generated_samples
close(hfile)