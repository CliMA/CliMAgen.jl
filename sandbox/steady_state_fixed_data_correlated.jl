using LinearAlgebra
using Statistics
using ProgressBars
using Flux
using CliMAgen
using BSON
using HDF5

using LinearAlgebra, Statistics

using Random
using SpeedyWeather
using StochasticStir
using SharedArrays

# load steady_data 
@info "Loading steady data"
hfile = h5open("steady_data_c.hdf5", "r")
timeseries = read(hfile["timeseries"])
μ = read(hfile, "shift")
σ = read(hfile, "scaling")
sigmax = read(hfile["sigmax"])
close(hfile)
hfile = h5open("steady_data_2_c.hdf5", "r")
timeseries2 = read(hfile["timeseries"])
close(hfile)
hfile = h5open("steady_data.hdf5", "r")
timeseries_u = read(hfile["timeseries"])
close(hfile)
hfile = h5open("steady_data_2.hdf5", "r")
timeseries2_u = read(hfile["timeseries"])
close(hfile)
@info "Loaded steady data"

## Define Score-Based Diffusion Model
@info "Defining Score Model"
FT = Float32
#Read in from toml
batchsize = 8
inchannels = 1
context_channels = 0 # length(my_fields)
sigma_min = FT(1e-3);
sigma_max = FT(sigmax);
nwarmup = 5000;
gradnorm = FT(1.0);
learning_rate = FT(2e-4);
beta_1 = FT(0.9);
beta_2 = FT(0.999);
epsilon = FT(1e-8);
ema_rate = FT(0.999);
device = Flux.gpu


# Create network
quick_arg = true
kernel_size = 3
kernel_sizes = [0, 0, 0, 0] # [3, 2, 1, 0]
channel_scale = 1
net = NoiseConditionalScoreNetwork(;
                                    channels = channel_scale .* [32, 64, 128, 256],
                                    proj_kernelsize   = kernel_size + kernel_sizes[1],
                                    outer_kernelsize  = kernel_size + kernel_sizes[2],
                                    middle_kernelsize = kernel_size + kernel_sizes[3],
                                    inner_kernelsize  = kernel_size + kernel_sizes[4],
                                    noised_channels = inchannels,
                                    context_channels = context_channels,
                                    context = false,
                                    shift_input = quick_arg,
                                    shift_output = quick_arg,
                                    mean_bypass = quick_arg,
                                    scale_mean_bypass = quick_arg,
                                    gnorm = quick_arg,
                                    )
score_model = VarianceExplodingSDE(sigma_max, sigma_min, net)
score_model = device(score_model)
score_model_smooth = deepcopy(score_model)
opt = Flux.Optimise.Optimiser(WarmupSchedule{FT}(nwarmup),
                              Flux.Optimise.ClipNorm(gradnorm),
                              Flux.Optimise.Adam(learning_rate,(beta_1, beta_2), epsilon)
) 
opt_smooth = ExponentialMovingAverage(ema_rate);
# model parameters
ps = Flux.params(score_model);
# setup smoothed parameters
ps_smooth = Flux.params(score_model_smooth);

#=
@info "Starting from checkpoint"
checkpoint_path = "steady_temperature_trunc_31.bson"# "checkpoint_large_temperature_vorticity_humidity_divergence_pressure_timestep.bson" # "checkpoint_large_temperature_vorticity_humidity_divergence_timestep_Base.RefValue{Int64}(10000).bson" # "checkpoint_large_temperature_vorticity_humidity_divergence_timestep.bson" # "checkpoint_large_temperature_vorticity_humidity_divergence_timestep_Base.RefValue{Int64}(130000).bson"
BSON.@load checkpoint_path model model_smooth opt opt_smooth
score_model = device(model)
score_model_smooth = device(model_smooth)
# model parameters
ps = Flux.params(score_model);
# setup smoothed parameters
ps_smooth = Flux.params(score_model_smooth);
=#


function lossfn_c(y; noised_channels = inchannels, context_channels=context_channels)
    x = y[:,:,1:noised_channels,:]
    # c = y[:,:,(noised_channels+1):(noised_channels+context_channels),:]
    return vanilla_score_matching_loss(score_model, x)
end

function generalization_loss(y; noised_channels = inchannels, context_channels=context_channels)
    x = y[:,:,1:noised_channels,:]
    # c = y[:,:,(noised_channels+1):(noised_channels+context_channels),:]
    N = size(x)[end]
    batchsize = 10
    M = N ÷ batchsize
    loss = 0.0
    for i in 1:M
        loss += vanilla_score_matching_loss(score_model_smooth, device(x[:,:,:,(1+(i-1)*batchsize):i*batchsize])) / (batchsize * M)
    end
    return loss
end

function mock_callback(batch; ps = ps, opt = opt, lossfn = lossfn_c, ps_smooth = ps_smooth, opt_smooth = opt_smooth)
    grad = Flux.gradient(() -> sum(lossfn(batch)), ps)
    Flux.Optimise.update!(opt, ps, grad)
    Flux.Optimise.update!(opt_smooth, ps_smooth, ps)
    return nothing
end

function one_epoch(y; noised_channels = inchannels, context_channels=context_channels)
    x = y[:,:,1:noised_channels,:]
    # c = y[:,:,(noised_channels+1):(noised_channels+context_channels),:]
    N = size(x)[end]
    shuffled_indices = shuffle(1:N)
    x = y[:,:,1:noised_channels, shuffled_indices]
    batchsize = 8
    M = N ÷ batchsize
    loss = 0.0
    for i in 1:M
        batch = device(x[:,:,:,(1+(i-1)*batchsize):i*batchsize])
        mock_callback(batch)
    end
    return nothing
end

# one epoch has 500 steps 
losses = Float64[]
losses2 = Float64[]
losses_u = Float64[]
losses2_u = Float64[]
for i in ProgressBar(1:1000)
    one_epoch(timeseries)
    loss1 = generalization_loss(timeseries)
    loss2 = generalization_loss(timeseries2)
    loss3 = generalization_loss(timeseries_u)
    loss4 = generalization_loss(timeseries2_u)
    push!(losses, loss1)
    push!(losses2, loss2)
    push!(losses_u, loss3)
    push!(losses2_u, loss4)
    println("Epoch $i: loss1 = $loss1, loss2 = $loss2")
    println("Epoch $i: loss3 = $loss3, loss4 = $loss4")
    println("---------")
end

hfile = h5open("losses_fixed_data_correlated.hdf5", "w")
hfile["losses"] = losses
hfile["losses_2"] = losses2
hfile["losses_u"] = losses_u
hfile["losses_2_u"] = losses2_u
close(hfile)

##

hfile = h5open("losses.hdf5", "r")
losses_online = read(hfile["losses"])
close(hfile)
##
losses[20]
losses_online[5]
##
using CairoMakie

fig = Figure()
ax = Axis(fig[1, 1]; title = "losses", xlabel ="epoch", ylabel = "loss")
lines!(ax, losses2_u, color = (:red,0.5), label = "loss fixed data")
lines!(ax, losses_online[5:5:end], color = (:blue, 0.5), label = "loss online training")
axislegend(ax, position = :rt)
xlims!(ax, 10, 500)
ylims!(ax, 0.0045 , losses2_u[end])
save("losses_fixed_vs_online_correlated.png", fig)

##
# In terms of gradient descent steps
fig = Figure()
ax = Axis(fig[1, 1]; title = "losses", xlabel ="1k batch updates", ylabel = "loss")
scale = 0.5
tmp = scale * (1:length(losses2_u))
tmp2 = scale * (1:length(losses_online[5:5:end]))
lines!(ax, tmp, losses2_u, color = (:red,0.5), label = "loss fixed data")
lines!(ax, tmp2, losses_online[5:5:end], color = (:blue, 0.5), label = "loss online training")
axislegend(ax, position = :rt)
xlims!(ax, scale * 10, scale * 500)
ylims!(ax, 0.0045 , losses2_u[end])
save("losses_fixed_vs_online_correlated_steps.png", fig)
