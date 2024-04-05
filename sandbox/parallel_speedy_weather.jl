using Distributed

@everywhere using Random
@everywhere using SpeedyWeather
@everywhere using StochasticStir
@everywhere using SharedArrays

@everywhere include("my_vorticity.jl")

spectral_grid = SpectralGrid(trunc=31, nlev=1)
my_vorticity_on_1 = MyInterpolatedVorticity(spectral_grid, schedule = Schedule(every=Day(1)))

gated_array = SharedArray{spectral_grid.NF}(my_vorticity_on_1.interpolator.locator.npoints, Distributed.nworkers())
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

nsteps = 2
const SLEEP_DURATION = 1e-2

@distributed for i in workers()
    id = myid()
    gate_id = id-1
    Random.seed!(1234+id)
    
    forcing = StochasticStirring(spectral_grid)
    drag = JetDrag(spectral_grid)
    model = ShallowWaterModel(;spectral_grid, forcing, drag)
    model.feedback.verbose = false
    
    my_vorticity = deepcopy(my_vorticity_on_1)
    add!(model.callbacks, my_vorticity)
    simulation = initialize!(model)
    
    for _ in 1:nsteps
        run!(simulation, period=Day(1))
        gate_written::Bool = false
        while ~gate_written
            if gate_open(gates, gate_id)
                gated_array[:, gate_id] .= my_vorticity.var
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
j = Ref(1)      # needs to be mutable somehow
while j[] <= nsteps
    if all_closed(gates)
        j[] += 1
        println(gated_array[1, :])
        open_all!(gates)
        # println("All gates opened.")
    else
        # println("Gates still open: $gates")
        sleep(SLEEP_DURATION)
    end
end