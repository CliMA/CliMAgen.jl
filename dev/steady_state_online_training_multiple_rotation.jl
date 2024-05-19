using LinearAlgebra
using Statistics
using ProgressBars
using Flux
using CliMAgen
using BSON
using HDF5

using Distributed
using LinearAlgebra, Statistics

if nworkers() < 8
    addprocs(8)
end

@everywhere using Random
@everywhere using SpeedyWeather
@everywhere using SharedArrays

@everywhere include("speedy_diagnostics.jl")
@everywhere include("speedy_weather.jl")
# trunc_val = 31
const trunc_val = 31
const add_pressure_field = false
fields =  [:temp_grid] 
layers = [5]
parameters = generate_parameters(; default=true)
simulation, my_fields = speedy_sim(; parameters, layers, fields, add_pressure_field)

gated_array = SharedArray{simulation.model.spectral_grid.NF}(my_fields[1].interpolator.locator.npoints, length(my_fields), Distributed.nworkers())
gated_array .= 0.0
# julia --project -p 8
# open is true, closed is false. Open means we can write to the array
gates = SharedVector{Bool}(nworkers())
open_all!(gates) = gates .= true
open_all!(gates)
@everywhere all_closed(gates) = sum(gates) == 0
@everywhere all_open(gates) = all(gates)
@everywhere gate_open(gates, gate_id::Integer) = gates[gate_id]
@everywhere close_gate!(gates, gate_id::Integer) = gates[gate_id] = false

# Initialize Speedy Weather Parameters
@info "initializing speedy weather sigma"
if myid() == 1
Random.seed!(1234) 

# sigma max
n_fields = length(my_fields)
# load steady_data 
@info "Loading steady data"
hfile = h5open("steady_default_data.hdf5", "r")
timeseries = read(hfile["timeseries"])
μ = read(hfile, "shift")
σ = read(hfile, "scaling")
sigmax = read(hfile["sigmax"])
close(hfile)
hfile = h5open("steady_default_data_correlated.hdf5", "r")
timeseries2 = read(hfile["timeseries"])
close(hfile)
@info "Loaded steady data"

## Define Score-Based Diffusion Model
@info "Defining Score Model"
FT = Float32
#Read in from toml
batchsize = nworkers()
inchannels = length(my_fields)
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

end # myid() == 1
##
@info "Done Defining score model"
# Run Models
nsteps = 20 * 10000 # 50 * 10000 # 10000 takes 1.5 hours
const SLEEP_DURATION = 1e-3

@distributed for i in workers()
    id = myid()
    gate_id = id-1
    Random.seed!(1998+id)
    # model
    fields =  [:temp_grid] 
    layers = [5]
    parameters = custom_parameters(; rotation = Float32((id-5.5)/3.5 * 1e-4))# generate_parameters(; default=true)
    simulation, my_fields = speedy_sim(; parameters, layers, fields, add_pressure_field)
    
    Nx, Ny = size(simulation.prognostic_variables.layers[1].timesteps[1].vor)
    simulation.prognostic_variables.layers[1].timesteps[1].vor .+= randn(Float32, Nx, Ny) * Float32(1e-10)
    run!(simulation, period=Day(1200))
    for (i, my_field) in enumerate(my_fields)
        gated_array[:, i, gate_id] .= my_field.var
    end
    for _ in 1:nsteps
        run!(simulation, period=Day(1))
        gate_written::Bool = false
        while ~gate_written
            if gate_open(gates, gate_id)
                for (i, my_field) in enumerate(my_fields)
                    gated_array[:, i, gate_id] .= my_field.var
                end
                close_gate!(gates, gate_id)
                gate_written = true
            else
                sleep(SLEEP_DURATION)
            end
        end
    end
    println("FINISHED.")
end

# HAPPENING ON PROC 1
if myid() == 1
    losses = Float64[]
    losses_2 = Float64[]
    tic = Base.time()
    j = Ref(1)      # needs to be mutable somehow
    while j[] <= nsteps
        if all_closed(gates)
            j[] += 1
            println(j)
            rbatch = copy(reshape(gated_array, (128, 64, length(my_fields), batchsize)))
            batch = (rbatch .- reshape(μ, (1, 1, length(my_fields), 1))) ./ reshape(σ, (1, 1, length(my_fields), 1))
            open_all!(gates)
            mock_callback(device(batch))
            if j[]%100 == 0
                loss = generalization_loss(timeseries)
                push!(losses, loss)
                loss2 = generalization_loss(timeseries2)
                push!(losses_2, loss2)
                @info "Loss at step $(j[]) is $loss and $loss2"
            end
            if j[]%40000 == 0 
                tmp = j[]
                @info "saving model"
                CliMAgen.save_model_and_optimizer(Flux.cpu(score_model), Flux.cpu(score_model_smooth), opt, opt_smooth, "checkpoint_steady_multiple_rotation_online_timestep_$tmp.bson")
            end
        else
            sleep(SLEEP_DURATION)
        end
    end
end

toc = Base.time()
println("Time for the simulation is $((toc-tic)/60) minutes.")