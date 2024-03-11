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

# run from giorgni2d
include("../utils_data.jl") # for data loading
include("../utils_analysis.jl") # for data loading
include("dataloader.jl") # for data loading

model_toml="Model.toml"
experiment_toml="Experiment.toml"
FT = Float32
toml_dict = TOML.parsefile(model_toml)
α = FT(toml_dict["param_group"]["alpha"])
β = FT(toml_dict["param_group"]["beta"])
γ = FT(toml_dict["param_group"]["gamma"])
σ = FT(toml_dict["param_group"]["sigma"])
f_path = "data/data_$(α)_$(β)_$(γ)_$(σ).hdf5"

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

train_dataloader, test_dataloader = get_data(f_path, "timeseries", nsamples)
x_dataloader = test_dataloader
train = getobs(test_dataloader.data, 1) # cat([x for x in x_dataloader]..., dims=4)

xtrain = train[:,:,1:noised_channels,:]
old_ctrain = train[:,:,(noised_channels+1):(noised_channels+context_channels),:]
ctrain = copy(old_ctrain)
nsamples = minimum([size(xtrain)[end], nsamples])
inds = 2:2
τ0 = reshape(time_embedding(1.0, size(xtrain)), (size(xtrain)[1],size(xtrain)[2], 1, 1)) 
if length(inds) == 1
    ctrain .= τ0 # draw from context 1:1
end
println("nsamples is ", nsamples)

checkpoint_path = joinpath(savedir, "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth
model = device(model_smooth)

# sample from the trained model
resolution = size(xtrain)[1]
time_steps, Δt, init_x = setup_sampler(
    model,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)
if sampler == "euler"
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:,:,:,1:nsamples])
    samples = cpu(samples) 
    samples2 = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=old_ctrain[:, :, :, 1:nsamples])
    samples2 = cpu(samples2)
elseif sampler == "pc"
    samples = predictor_corrector_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples])
    samples = cpu(samples) 
    samples2 = predictor_corrector_sampler(model, init_x, time_steps, Δt; c=old_ctrain[:, :, :, 1:nsamples])
    samples2 = cpu(samples2)
end


spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png")
qq_plot(xtrain[:, :, :, 1:nsamples], samples, savedir, "qq_plot.png")
spectrum_plot(xtrain[:, :, :, 1:nsamples], samples, savedir, "mean_spectra.png")

# create plots with nimages images of sampled data and training data
for ch in 1:inchannels
    heatmap_grid(samples[:, :, [ch], 1:nimages], 1, savedir, "$(sampler)_images_$(ch).png")
    heatmap_grid(xtrain[:, :, [ch], 1:nimages], 1, savedir, "train_images_$(ch).png")
end
ncum = 10
cum_x = zeros(ncum)
cum_samples = zeros(ncum)
for i in 1:ncum
    cum_x[i] = cumulant(xtrain[:],i)
    cum_samples[i] = cumulant(samples[:],i)
end
scatter(cum_x,label="Data",xlabel="Cumulants")
scatter!(cum_samples, label="Gen")
savefig(joinpath(savedir,"cumulants.png"))

stephist(xtrain[:],normalize=:pdf,label="Data")
stephist!(samples[:],normalize=:pdf, label="Gen")
savefig(joinpath(savedir,"pdfs.png"))

loss_plot(savedir, "losses.png"; xlog = false, ylog = true)

hfile = h5open(f_path[1:end-5] * "_analysis.hdf5", "w")
hfile["data cumulants"] = cum_x
hfile["generative cumulants"] = cum_samples
hfile["samples"] = samples
hfile["data"] = xtrain
hfile["context"] = ctrain
hfile["context indices"] = collect(inds)
hfile["samples with various conditionals"] = samples2
hfile["contexts"] = old_ctrain
close(hfile)


#=
tmp = test_dataloader.data.data.data
using GLMakie 
fig = Figure() 
ax1 = GLMakie.Axis(fig[1,1])
GLMakie.heatmap!(ax1, tmp[:,:,1,1])
ax2 = GLMakie.Axis(fig[1,2])
GLMakie.heatmap!(ax2, tmp[:,:,1,1 + 128])
display(fig)
=#