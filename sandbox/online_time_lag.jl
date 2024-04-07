using LinearAlgebra
using Statistics
using ProgressBars
using Flux
using CliMAgen
using BSON
import CliMAgen: GaussianFourierProjection

using Distributed
using LinearAlgebra, Statistics

if nworkers() < 8
    addprocs(8)
end

@everywhere using Random
@everywhere using SpeedyWeather
@everywhere using StochasticStir
@everywhere using SharedArrays

@everywhere include("my_vorticity.jl")

spectral_grid = SpectralGrid(trunc=31, nlev=1)
my_vorticity_on_1 = MyInterpolatedVorticity(spectral_grid, schedule = Schedule(every=Day(1)))

gated_array = SharedArray{spectral_grid.NF}(my_vorticity_on_1.interpolator.locator.npoints, 2, Distributed.nworkers())
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
forcing = StochasticStirring(spectral_grid)
drag = JetDrag(spectral_grid)
speedy_model = ShallowWaterModel(;spectral_grid, forcing, drag)
speedy_model.feedback.verbose = false

my_vorticity = deepcopy(my_vorticity_on_1)
add!(speedy_model.callbacks, my_vorticity)
simulation = initialize!(speedy_model)
run!(simulation, period=Day(30))
tmp1 = copy(my_vorticity.var)
run!(simulation, period=Day(30))
tmp2 = copy(my_vorticity.var)

rtmp1 = reshape(tmp1, (128, 64))
rtmp2 = reshape(tmp2, (128, 64))
atmp1 = (rtmp1[1:2:end, :] + rtmp1[2:2:end, :])/2
atmp2 = (rtmp2[1:2:end, :] + rtmp2[2:2:end, :])/2
σ = quantile(abs.(atmp2[:]), 0.9)
sigmax = norm(atmp1 - atmp2) / σ * 1.2
##
function GaussianFourierProjection(embed_dim::Int, embed_dim2::Int, scale::FT) where {FT}
    Random.seed!(1234) # same thing every time
    W = randn(FT, embed_dim ÷ 2, embed_dim2) .* scale
    return CliMAgen.GaussianFourierProjection(W)
end

gfp = GaussianFourierProjection(64, 64, 30.0f0)
cpu_batch = zeros(Float32, 64, 64, 3, nworkers())
halfworkers = nworkers() ÷ 2
cpu_batch[:, :, 3, 1:halfworkers] .= gfp(0.0f0)
cpu_batch[:, :, 3, halfworkers+1:end] .= gfp(1.0f0)
## Define Score-Based Diffusion Model
FT = Float32
#Read in from toml
batchsize = nworkers()
inchannels = 1
sigma_min = FT(1e-2);
sigma_max = FT(sigmax);
context_channels=2
context = true
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
#=
net = NoiseConditionalScoreNetwork(;
                                    noised_channels = inchannels,
                                    context_channels = context_channels,
                                    context = true,
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
=#

@info "Starting from checkpoint"
checkpoint_path = "checkpoint_online_time_lag.bson"
# BSON.@load checkpoint_path score_model score_model_smooth opt opt_smooth
BSON.@load checkpoint_path model model_smooth opt opt_smooth
score_model = device(model)
score_model_smooth = device(model_smooth)
# model parameters
ps = Flux.params(score_model);
# setup smoothed parameters
ps_smooth = Flux.params(score_model_smooth);


function lossfn_c(y; noised_channels = inchannels, context_channels=context_channels)
    x = y[:,:,1:noised_channels,:]
    c = y[:,:,(noised_channels+1):(noised_channels+context_channels),:]
    return score_matching_loss(score_model, x; c = c)
end

function mock_callback(batch; ps = ps, opt = opt, lossfn = lossfn_c, ps_smooth = ps_smooth, opt_smooth = opt_smooth)
    grad = Flux.gradient(() -> sum(lossfn(batch)), ps)
    Flux.Optimise.update!(opt, ps, grad)
    Flux.Optimise.update!(opt_smooth, ps_smooth, ps)
    return nothing
end

end # myid() == 1
##

# Run Models
base_steps = 1200 # five minutes
hour_steps = 12 * base_steps
nsteps = 3 * hour_steps #140 * 10000 takes 11 hours
const SLEEP_DURATION = 1e-2

@distributed for i in workers()
    id = myid()
    gate_id = id-1
    Random.seed!(1234+id)
    
    forcing = StochasticStirring(spectral_grid)
    drag = JetDrag(spectral_grid)
    speedy_model = ShallowWaterModel(;spectral_grid, forcing, drag)
    speedy_model.feedback.verbose = false
    
    my_vorticity = deepcopy(my_vorticity_on_1)
    add!(speedy_model.callbacks, my_vorticity)
    simulation = initialize!(speedy_model)
    run!(simulation, period=Day(30))
    gated_array[:, 1, gate_id] .= my_vorticity.var
    for _ in 1:nsteps
        run!(simulation, period=Day(1))
        gate_written::Bool = false
        while ~gate_written
            if gate_open(gates, gate_id)
                if gate_id ≤ halfworkers
                    gated_array[:, 2, gate_id] .= my_vorticity.var
                else
                    gated_array[:, 2, gate_id] .= gated_array[:, 1, gate_id]
                end
                # gated_array[:, 2, gate_id] .= gated_array[:, 1, gate_id]
                gated_array[:, 1, gate_id] .= my_vorticity.var
                close_gate!(gates, gate_id)
                # println("Closing gate $gate_id")
                gate_written = true
            else
                # println("Gate $gate_id closed, sleep.")
                sleep(SLEEP_DURATION)
            end
        end
    end
    println("FINISHED.")
end

# HAPPENING ON PROC 1
if myid() == 1
    tic = Base.time()
    j = Ref(1)      # needs to be mutable somehow
    while j[] <= nsteps
        if all_closed(gates)
            j[] += 1
            # println(gated_array[1, :])
            println(j)
            
            rbatch = copy(reshape(gated_array, (128, 64, 2, batchsize)));
            cpu_batch[:, :, 1:2, :] .= (rbatch[1:2:end, :, :, :] + rbatch[2:2:end, :, :, :]) / (2σ);
            open_all!(gates)
            mock_callback(device(cpu_batch))
            # mock_callback(device(batch))
            # println("All gates opened.")
        else
            # println("Gates still open: $gates")
            sleep(SLEEP_DURATION)
        end
    end
end

toc = Base.time()
println("Time for the simulation is $((toc-tic)/60) minutes.")

CliMAgen.save_model_and_optimizer(Flux.cpu(score_model), Flux.cpu(score_model_smooth), opt, opt_smooth, "checkpoint_online_time_lag_3.bson")