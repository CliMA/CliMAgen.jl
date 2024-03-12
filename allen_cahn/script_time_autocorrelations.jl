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
Random.seed!(1234)
# run from giorgni2d
include("utils_analysis.jl") # for data loading
include("dataloader.jl") # for data loading

model_toml = "ModelTimeseries.toml"
experiment_toml = "ExperimentTimeseries.toml"
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

hfile = h5open("data/data_1.0_0.0_0.0_0.0.hdf5", "r")
tseries = read(hfile, "timeseries");
close(hfile)
loader_train = DataLoaders.DataLoader(TimeseriesData(tseries), nsamples)
loader_test = DataLoaders.DataLoader(TimeseriesData(tseries), nsamples)

dataloaders = (; loader_train, loader_test)
x_dataloader = loader_test
train = getobs(loader_test.data, 1) 

tmp_shuffle = shuffle(1:nsamples)
xtrain = train[:, :, 1:noised_channels, tmp_shuffle]
old_ctrain = train[:, :, (noised_channels+1):(noised_channels+context_channels), tmp_shuffle]
ctrain = copy(old_ctrain)
nsamples = minimum([size(xtrain)[end], nsamples])
# τ0 = # reshape(time_embedding(1.0, size(xtrain)), (size(xtrain)[1], size(xtrain)[2], 1, 1))
t = 30/1000
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, 1, :] .= τ0 # draw from context 1:1
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


lags = collect(0:0.25:60)
ai_autocorrelations = zeros(length(lags));
rtseries = reshape(tseries, (32, 32, 128, 2000));
for (j,t) in ProgressBar(enumerate(lags))
    τ0 = reshape(gfp(t/1000), (size(xtrain)[1], size(xtrain)[2], 1, 1))
    ctrain[:, :, 1, 1:nsamples] .= τ0 # draw from context 1:1
    samples = cpu(Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples]))
    ai_autocorrelations[j] = mean(samples[:, :, 1, :] .* samples[:, :, 2, :]) - mean(samples[:, :, 1, :]) * mean(samples[:, :, 2, :])
end


data_autocorrelations = zeros(length(0:lags[end]));
for (j, t) in ProgressBar(enumerate(0:round(Int,lags[end])))
    data_autocorrelations[j] = mean(rtseries[:, :, 1:nsamples, 1:end-t] .* rtseries[:, :, 1:nsamples, 1+t:end]) - mean(rtseries[:, :, 1:nsamples, 1:end-t]) * mean(rtseries[:, :, 1:nsamples, 1+t:end])
end
##
using CairoMakie

fig = Figure()
lw = 6
ax = CairoMakie.Axis(fig[1,1]; xlabel = "Lag", ylabel = "Autocorrelation", title = "raw")
CairoMakie.scatter!(ax, lags, ai_autocorrelations , color=(:blue, 0.5), label="AI", linewidth = lw)
CairoMakie.scatter!(ax, 0:lags[end], data_autocorrelations, color=(:red, 0.5), label="Data", linewidth = lw)
# CairoMakie.ylims!(ax, 0.0, data_autocorrelations[1] * 1.1)
CairoMakie.axislegend(ax)
CairoMakie.save(pwd() * "/autocorrelations.png", fig)

fig = Figure()
ax = CairoMakie.Axis(fig[1,1]; xlabel = "Lag", ylabel = "Autocorrelation", title = "normalized")
CairoMakie.scatter!(ax, lags, ai_autocorrelations / ai_autocorrelations[1], color=(:blue, 0.5), label="AI", linewidth = lw)
CairoMakie.scatter!(ax, 0:lags[end], data_autocorrelations / data_autocorrelations[1], color=(:red, 0.5), label="Data", linewidth = lw)
# CairoMakie.ylims!(ax, -0.1, 1.1)
CairoMakie.axislegend(ax)
CairoMakie.save(pwd() * "/normalized_autocorrelations.png", fig)