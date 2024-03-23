nsteps = 4000
const SLEEP_DURATION = 1e-3

#=
cpu_batch[:, :, 3, 1:halfworkers] .= gfp(0.0f0)
cpu_batch[:, :, 3, halfworkers+1:end] .= gfp(1.0f0)
=#

# cpu_batch[:, :, 3, :] .= gfp(1.0f0)

tic = Base.time()
@distributed for i in workers()
    id = myid()
    gate_id = id-1
    Random.seed!(1235+id)
    
    forcing = StochasticStirring(spectral_grid)
    drag = JetDrag(spectral_grid)
    model = ShallowWaterModel(;spectral_grid, forcing, drag)
    model.feedback.verbose = false
    
    my_vorticity = deepcopy(my_vorticity_on_1)
    add!(model.callbacks, my_vorticity)
    simulation = initialize!(model)
    run!(simulation, period=Day(30))
    gated_array[:, 1, gate_id] .= my_vorticity.var
    for _ in 1:nsteps
        run!(simulation, period=Day(1))
        gate_written::Bool = false
        while ~gate_written
            if gate_open(gates, gate_id)
                gated_array[:, 2, gate_id] .= gated_array[:, 1, gate_id]
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