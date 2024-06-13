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

# load steady_data 
@info "Loading steady data"
hfile = h5open("steady_default_data_correlated.hdf5", "r")
timeseries = read(hfile["timeseries"])
μ = read(hfile, "shift")
σ = read(hfile, "scaling")
sigmax = read(hfile["sigmax"])
close(hfile)
hfile = h5open("steady_default_data.hdf5", "r")
timeseries2 = read(hfile["timeseries"])
close(hfile)
@info "Loaded steady data"
hfile = h5open("steady_default_data_correlated_part_2.hdf5", "r")
timeseries3 = read(hfile["timeseries"])
close(hfile)
hfile = h5open("steady_default_data_correlated_part_3.hdf5", "r")
timeseries4 = read(hfile["timeseries"])
close(hfile)
timeseries = cat(timeseries, timeseries4, dims = 4)


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
println(size(timeseries))
losses = Float64[]
losses2 = Float64[]
losses_3 = Float64[]
losses_4 = Float64[]
for i in ProgressBar(1:800)
    one_epoch(timeseries)
    loss1 = generalization_loss(timeseries)
    loss2 = generalization_loss(timeseries2)
    loss3 = generalization_loss(timeseries3)
    loss4 = generalization_loss(timeseries4)
    push!(losses, loss1)
    push!(losses2, loss2)
    push!(losses_3, loss3)
    push!(losses_4, loss4)
    println("Epoch $i: loss1 = $loss1, loss2 = $loss2")
    if i%100 == 0
        @info "saving epoch $i"
        println("Epoch $i: loss1 = $loss1, loss2 = $loss2")
        CliMAgen.save_model_and_optimizer(Flux.cpu(score_model), Flux.cpu(score_model_smooth), opt, opt_smooth, "steady_state_more_fixed_data_epoch_$i.bson")
    end
end

CliMAgen.save_model_and_optimizer(Flux.cpu(score_model), Flux.cpu(score_model_smooth), opt, opt_smooth, "steady_state_fixed_data.bson")

hfile = h5open("losses_more_fixed_data.hdf5", "w")
hfile["losses"] = losses
hfile["losses_2"] = losses2
hfile["losses_3"] = losses_3
hfile["losses_4"] = losses_4
close(hfile)