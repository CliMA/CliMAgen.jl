using LinearAlgebra
using Statistics
using ProgressBars
using Flux
using CliMAgen

using Distributed
using LinearAlgebra, Statistics

if nworkers() < 8
    addprocs(8)
end

@everywhere using Random
@everywhere using SpeedyWeather
@everywhere using StochasticStir
@everywhere using SharedArrays

@everywhere include("my_field.jl")
fields =  [:temp_grid, :vor_grid, :humid_grid, :div_grid]
layers = [1]
spectral_grid = SpectralGrid(trunc=31, nlev=5)

my_fields = []
for field in fields, layer in layers
    my_field_on_1 = MyInterpolatedField(spectral_grid; schedule = Schedule(every=Day(1)), field_name = field, layer = layer)
    my_field = deepcopy(my_field_on_1)
    push!(my_fields, my_field)
end

gated_array = SharedArray{spectral_grid.NF}(my_fields[1].interpolator.locator.npoints, length(my_fields), Distributed.nworkers())
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

# model
ocean = AquaPlanet(spectral_grid, temp_equator=302, temp_poles=273)
land_sea_mask = AquaPlanetMask(spectral_grid)
orography = NoOrography(spectral_grid)
model = PrimitiveWetModel(; spectral_grid, ocean) 
model.feedback.verbose = false
# callbacks
for my_field in my_fields 
    add!(model.callbacks, my_field)
end
# run
simulation = initialize!(model)
run!(simulation, period=Day(100))
tmp1 = zeros(spectral_grid.NF, my_fields[1].interpolator.locator.npoints, length(fields) * length(layers))
for (i, my_field) in enumerate(my_fields)
    tmp1[:, i] = copy(my_fields[i].var)
end
run!(simulation, period=Day(100))
tmp2 = zeros(spectral_grid.NF, my_fields[1].interpolator.locator.npoints, length(fields) * length(layers))
for (i, my_field) in enumerate(my_fields)
    tmp2[:, i] = copy(my_fields[i].var)
end
# sigma max
n_fields = length(fields) * length(layers)
rtmp1 = reshape(tmp1, (128, 64, n_fields)) 
rtmp2 = reshape(tmp2, (128, 64, n_fields))
μ = mean((rtmp1 + rtmp2)/2, dims = (1, 2))
σ = reshape([quantile(abs.((rtmp2[:, :, i] + rtmp1[:,:,i])/2 .- μ[i])[:], 0.95) for i in 1:n_fields], (1, 1, n_fields))
sigmax = norm((rtmp1 - rtmp2) ./ σ)  * 1.2

## Define Score-Based Diffusion Model
@info "Defining Score Model"
FT = Float32
#Read in from toml
batchsize = nworkers()
inchannels = length(my_fields)
sigma_min = FT(1e-2);
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
net = NoiseConditionalScoreNetwork(;
                                    noised_channels = inchannels,
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
lossfn = x -> score_matching_loss(score_model, x);
function mock_callback(batch; ps = ps, opt = opt, lossfn = lossfn, ps_smooth = ps_smooth, opt_smooth = opt_smooth)
    grad = Flux.gradient(() -> sum(lossfn(batch)), ps)
    Flux.Optimise.update!(opt, ps, grad)
    Flux.Optimise.update!(opt_smooth, ps_smooth, ps)
    return nothing
end

end # myid() == 1
##
@info "Done Defining score model"
# Run Models
nsteps = 10 * 10000 # 10000 takes 1.5 hours
const SLEEP_DURATION = 1e-3

@distributed for i in workers()
    id = myid()
    gate_id = id-1
    Random.seed!(1234+id)
    # model
    ocean = AquaPlanet(spectral_grid, temp_equator=302, temp_poles=273)
    land_sea_mask = AquaPlanetMask(spectral_grid)
    orography = NoOrography(spectral_grid)
    model = PrimitiveWetModel(; spectral_grid, ocean)
    model.feedback.verbose = false
    # callbacks
    my_fields = []
    for field in fields, layer in layers
        my_field_on_1 = MyInterpolatedField(spectral_grid; schedule = Schedule(every=Day(1)), field_name = field, layer = layer)
        my_field = deepcopy(my_field_on_1)
        push!(my_fields, my_field)
        add!(model.callbacks, my_field)
    end
    # initialize and run
    simulation = initialize!(model)
    run!(simulation, period=Day(100))
    
    for _ in 1:nsteps
        run!(simulation, period=Day(1))
        gate_written::Bool = false
        while ~gate_written
            if gate_open(gates, gate_id)
                for (i, my_field) in enumerate(my_fields)
                    gated_array[:, i, gate_id] .= my_field.var
                end
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
            rbatch = copy(reshape(gated_array, (128, 64, length(my_fields), batchsize)))
            batch = (rbatch .- reshape(μ, (1, 1, length(my_fields), 1))) ./ reshape(σ, (1, 1, length(my_fields), 1))
            open_all!(gates)
            mock_callback(device(batch))
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