using ProgressBars, LinearAlgebra, HDF5

include("speedy_weather.jl")

add_pressure_field = false
fields =  [:temp_grid] 
layers = [5]
parameters = generate_parameters(; default=true)
simulation, my_fields = speedy_sim(; parameters, layers, fields, add_pressure_field)

@info "Run for 1200 days"
run!(simulation, period=Day(1200))

@info "Gather Timeseries"
steps = 1000 # 4000
skip_days = 14 # 14
timeseries = zeros(simulation.model.spectral_grid.NF, my_fields[1].interpolator.locator.npoints, length(my_fields), steps)
for j in ProgressBar(1:steps)
    run!(simulation, period=Day(skip_days))
    for (i, my_field) in enumerate(my_fields)
        timeseries[:, i, j] = copy(my_fields[i].var)
    end
end


n_fields = length(my_fields)
r_timeseries = copy(reshape(timeseries, (128, 64, n_fields, steps)) )
μ = mean(r_timeseries, dims = (1, 2, 4))
μ_drift = mean(r_timeseries, dims = (1, 2))
σ = reshape([quantile(abs.(r_timeseries[:, :, i, :] .- μ)[:], 0.999) for i in 1:n_fields], (1, 1, n_fields, 1))

rescaled_timeseries = (r_timeseries .- μ) ./ σ

shuffled_steps = shuffle(1:steps)
N = minimum([steps, 100])
shuffled_steps_1 = shuffled_steps[1:N]
shuffled_steps_2 = shuffled_steps[end-N+1:end]
distances = [norm(rescaled_timeseries[:,:,:,i]-rescaled_timeseries[:,:,:,j]) for i in shuffled_steps_1, j in shuffled_steps_2]
sigmax = maximum(distances) * 1.2

lat, lon = RingGrids.get_latdlonds(my_fields[1].var);
lon = reshape(lon, (128, 64))[:,1];
lat = reshape(lat, (128, 64))[1, :];


hfile = h5open("steady_default_data.hdf5", "w")
hfile["timeseries"] = rescaled_timeseries
hfile["shift"] = μ
hfile["scaling"] = σ
hfile["sigmax"] = sigmax
hfile["lat"] = lat
hfile["lon"] = lon
close(hfile)


simulation2 = deepcopy(simulation)

@info "Gather Timeseries Part 2"
steps = 4000 # 4000
skip_days = 1 # 14
timeseries = zeros(simulation.model.spectral_grid.NF, my_fields[1].interpolator.locator.npoints, length(my_fields), steps)
for j in ProgressBar(1:steps)
    run!(simulation, period=Day(skip_days))
    for (i, my_field) in enumerate(my_fields)
        timeseries[:, i, j] = copy(my_fields[i].var)
    end
end


r_timeseries = copy(reshape(timeseries, (128, 64, n_fields, steps)) )
rescaled_timeseries = (r_timeseries .- μ) ./ σ

hfile = h5open("steady_default_data_correlated.hdf5", "w")
hfile["timeseries"] = rescaled_timeseries
hfile["shift"] = μ
hfile["scaling"] = σ
hfile["sigmax"] = sigmax
hfile["lat"] = lat
hfile["lon"] = lon
close(hfile)

