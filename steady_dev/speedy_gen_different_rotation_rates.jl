using ProgressBars, LinearAlgebra, HDF5

include("speedy_weather.jl")

add_pressure_field = false
fields =  [:temp_grid] 
layers = [5]

spinup = 365 * 100
steps = 10000 # 4000
skip_days = 1 # 14
rotation_rates = [Float32(0.6e-4), Float32(1.1e-4), Float32(1.5e-4), Float32(7.29e-5)]

for rotation_index in ProgressBar(1:4)
    # rotation_index = 4
    @info "Rotation index $rotation_index"
    @info "Run for $spinup days"
    rotation = rotation_rates[rotation_index]
    parameters = custom_parameters(; rotation)
    simulation, my_fields = speedy_sim(; parameters, layers, fields, add_pressure_field)
    run!(simulation, period=Day(spinup))

    @info "Gather Timeseries"
    timeseries = zeros(simulation.model.spectral_grid.NF, my_fields[1].interpolator.locator.npoints, length(my_fields), steps)
    for j in ProgressBar(1:steps)
        run!(simulation, period=Day(skip_days))
        for (i, my_field) in enumerate(my_fields)
            timeseries[:, i, j] = copy(my_fields[i].var)
        end
    end

    hfile = h5open("rotation_rate_data_$(rotation_index).hdf5", "w")
    hfile["timeseries"] = timeseries 
    hfile["rotation"] = rotation 
    close(hfile)
end


