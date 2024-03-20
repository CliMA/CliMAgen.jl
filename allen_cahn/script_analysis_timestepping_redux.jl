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
using CairoMakie
Random.seed!(1234)
# run from giorgni2d
include("utils_analysis.jl") # for data loading
include("dataloader.jl") # for data loading

model_toml = "ModelTimestepping.toml"
experiment_toml = "ExperimentTimestepping.toml"
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
ctrain = Float32.(copy(old_ctrain))
#=
nsamples = minimum([size(xtrain)[end], nsamples])
# τ0 = # reshape(time_embedding(1.0, size(xtrain)), (size(xtrain)[1], size(xtrain)[2], 1, 1))
t = 0.0/300# 0/1000
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, 1, :] .= τ0 # draw from context 1:1
=#

nsamples = 1
println("nsamples is ", nsamples)
tlag = 0
t = tlag/60
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, [2], 1:nsamples] .= τ0

checkpoint_path = joinpath(savedir, "checkpoint_save.bson")
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
rng = MersenneTwister(1234)



tlag = 0
t = tlag/60
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, [2], 1:nsamples] .= τ0

nsteps = 10
us = zeros(Float32, 32, 32, nsteps);
us[:, :, 1] .= ctrain[:, :, [1], 1:nsamples]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = CairoMakie.Axis(fig[ii,jj]; title = "t-lag = $(tlag * i), iteration = $i")
    CairoMakie.heatmap!(ax, us[:,:,i], colorrange = (-0.5, 0.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag0.png", fig)


tlag = 30
t = tlag/60
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, [2], 1:nsamples] .= τ0

nsteps = 34
us = zeros(Float32, 32, 32, nsteps);
us[:, :, 1] .= ctrain[:, :, [1], 1:nsamples]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = CairoMakie.Axis(fig[jj, ii]; title = "t-lag = $(tlag * i), iteration = $i")
    CairoMakie.heatmap!(ax, us[:,:,10-i], colorrange = (-0.5, 0.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag30.png", fig)

fig = Figure() 
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.hist!(ax, us[:,:,end][:], normalization = :pdf, bins = 20)
save("histogram_lag30.png", fig)

autocov = reverse([mean(us[:,:,end] .* us[:,:,i]) - mean(us[:,:,1]) .* mean(us[:,:,i]) for i in 1:nsteps])
fig = Figure() 
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.lines!(ax, collect(0:nsteps-1) .* tlag, autocov, label = "autocovariance")
CairoMakie.ylims!(ax, -0.01, 0.05)
save("autocov_time_lag30.png", fig)


##

tlag = 10
t = tlag/60
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, [2], 1:nsamples] .= τ0

nsteps = 100
us = zeros(Float32, 32, 32, nsteps);
us[:, :, 1] .= ctrain[:, :, [1], 1:nsamples]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = CairoMakie.Axis(fig[jj, ii]; title = "t-lag = $(tlag * i), iteration = $i")
    CairoMakie.heatmap!(ax, us[:,:,10-i], colorrange = (-0.5, 0.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag10.png", fig)

fig = Figure() 
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.hist!(ax, us[:,:,end][:], normalization = :pdf, bins = 20)
save("histogram_lag10.png", fig)

autocov = reverse([mean(us[:,:,end] .* us[:,:,i]) - mean(us[:,:,1]) .* mean(us[:,:,i]) for i in 1:nsteps])
fig = Figure() 
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.lines!(ax, collect(0:nsteps-1) .* tlag, autocov, label = "autocovariance")
CairoMakie.ylims!(ax, -0.01, 0.05)
save("autocov_time_lag10.png", fig)


##
tlag = 60
t = tlag/60
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, [2], 1:nsamples] .= τ0

nsteps = 17
us = zeros(Float32, 32, 32, nsteps);
us[:, :, 1] .= ctrain[:, :, [1], 1:nsamples]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = CairoMakie.Axis(fig[jj, ii]; title = "t-lag = $(tlag * i), iteration = $i")
    CairoMakie.heatmap!(ax, us[:,:,10-i], colorrange = (-0.5, 0.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag60.png", fig)

fig = Figure() 
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.hist!(ax, us[:,:,end][:], normalization = :pdf, bins = 20)
save("histogram_lag60.png", fig)

autocov = reverse([mean(us[:,:,end] .* us[:,:,i]) - mean(us[:,:,1]) .* mean(us[:,:,i]) for i in 1:nsteps])
fig = Figure() 
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.lines!(ax, collect(0:nsteps-1) .* tlag, autocov, label = "autocovariance")
CairoMakie.ylims!(ax, -0.01, 0.05)
save("autocov_time_lag60.png", fig)



##
tlag = 10
t = tlag/60
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, [2], 1:nsamples] .= τ0

nsteps = 2000
us = zeros(Float32, 32, 32, nsteps);
us[:, :, 1] .= ctrain[:, :, [1], 1:nsamples]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = CairoMakie.Axis(fig[jj, ii]; title = "t-lag = $(tlag * i), iteration = $i")
    CairoMakie.heatmap!(ax, us[:,:,end-i], colorrange = (-0.5, 0.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag10_long.png", fig)

autocov = reverse([mean(us[:,:,end] .* us[:,:,i]) - mean(us[:,:,1]) .* mean(us[:,:,i]) for i in 1:nsteps])
fig = Figure() 
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.lines!(ax, collect(0:nsteps-1) .* tlag, autocov, label = "autocovariance")
CairoMakie.ylims!(ax, -0.01, 0.05)
save("autocov_time_lag10_long.png", fig)

rus = copy(us[:,:,reverse(1:size(us)[end])])

autov2 = [mean(us[:,:,1:end-i] .* us[:,:,1+i:end]) - mean(us[:,:,1+i:end]) .* mean(us[:,:,1:end-i]) for i in 0:100]


rtseries = reshape(tseries, (32, 32, 128, 2000))[:, :, 1:10, :];
autovtrue = [mean(rtseries[:,:,:, 1:2000-i] .* rtseries[:,:,:, 1+i:2000]) - mean(rtseries[:,:,:,1+i:2000]) .* mean(rtseries[:,:,:, 1:2000-i]) for i in 0:1000];

fig = Figure() 
ax = CairoMakie.Axis(fig[1,1])
CairoMakie.lines!(ax, collect(0:100) .* tlag, autov2, label = "autocovariance ai", color = (:blue, 0.5), linewidth = 5)
CairoMakie.lines!(ax, collect(0:1000) , autovtrue, label = "autocovariance true", color = (:red, 0.5), linewidth = 5)
axislegend(ax, position = :rt)
CairoMakie.ylims!(ax, -0.01, 0.05)
save("autocov_time_lag10_long_morestats.png", fig)

##
u0 = randn(Float32, 32, 32, 1, 1)
nsteps = 10
us = randn(Float32, 32, 32, nsteps);
us[:, :, 1] .= u0
ctrain[:, :, [1], [1]] .= u0
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end


fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = CairoMakie.Axis(fig[jj, ii]; title = "t-lag = $(tlag * i), iteration = $i")
    CairoMakie.heatmap!(ax, us[:,:, i], colorrange = (-0.5, 0.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag10_long_random_start.png", fig)


u0 = ones(Float32, 32, 32, 1, 1)
nsteps = 10
us = randn(Float32, 32, 32, nsteps);
us[:, :, 1] .= u0
ctrain[:, :, [1], [1]] .= u0
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end


fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = CairoMakie.Axis(fig[jj, ii]; title = "t-lag = $(tlag * i), iteration = $i")
    CairoMakie.heatmap!(ax, us[:,:, i], colorrange = (-0.5, 0.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag10_long_one_start.png", fig)

##


u0 = zeros(Float32, 32, 32, 1, 1)
nsteps = 10
us = randn(Float32, 32, 32, nsteps);
us[:, :, 1] .= u0
ctrain[:, :, [1], [1]] .= u0
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end


fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = CairoMakie.Axis(fig[jj, ii]; title = "t-lag = $(tlag * i), iteration = $i")
    CairoMakie.heatmap!(ax, us[:,:, i], colorrange = (-0.5, 0.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag10_long_zero_start.png", fig)