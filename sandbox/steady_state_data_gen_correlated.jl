using LinearAlgebra
using Statistics
using ProgressBars
using Flux
using CliMAgen
using BSON
using LinearAlgebra, Statistics

using Random
using SpeedyWeather
using StochasticStir
using SharedArrays
using HDF5


include("my_field.jl")
include("my_pressure.jl")

const trunc_val = 31
const add_pressure_field = false

fields =  [:temp_grid] # [:temp_grid, :vor_grid, :humid_grid, :div_grid]
layers = [1] #  [1, 2, 3, 4, 5]
spectral_grid = SpectralGrid(trunc=trunc_val, nlev=5)

my_fields = []
for layer in layers, field in fields
    my_field_on_1 = MyInterpolatedField(spectral_grid; schedule = Schedule(every=Day(1)), field_name = field, layer = layer)
    my_field = deepcopy(my_field_on_1)
    push!(my_fields, my_field)
end
if add_pressure_field
    my_field = MyInterpolatedPressure(spectral_grid; schedule = Schedule(every=Day(1)))
    push!(my_fields, my_field)
end

# Initialize Speedy Weather Parameters
@info "initializing speedy weather sigma"
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
@info "Run for 100 days"
run!(simulation, period=Day(100))
@info "Gather Timeseries"
steps = 4000
skip_days = 1
timeseries = zeros(spectral_grid.NF, my_fields[1].interpolator.locator.npoints, length(my_fields), steps)
for j in ProgressBar(1:steps)
    run!(simulation, period=Day(skip_days))
    for (i, my_field) in enumerate(my_fields)
        timeseries[:, i, j] = copy(my_fields[i].var)
    end
end

n_fields = length(my_fields)
r_timeseries = copy(reshape(timeseries, (128, 64, n_fields, steps)) )

hfile = h5open("steady_data.hdf5", "r")
μ = read(hfile, "shift")
σ = read(hfile, "scaling")
sigmax = read(hfile["sigmax"])
close(hfile)

rescaled_timeseries = (r_timeseries .- μ) ./ σ

##
hfile = h5open("steady_data_c.hdf5", "w")
hfile["timeseries"] = rescaled_timeseries
hfile["shift"] = μ
hfile["scaling"] = σ
hfile["sigmax"] = sigmax
close(hfile)

##
@info "Gather Timeseries Part 2"
timeseries = zeros(spectral_grid.NF, my_fields[1].interpolator.locator.npoints, length(my_fields), steps)
for j in ProgressBar(1:steps)
    run!(simulation, period=Day(skip_days))
    for (i, my_field) in enumerate(my_fields)
        timeseries[:, i, j] = copy(my_fields[i].var)
    end
end

n_fields = length(my_fields)
r_timeseries = copy(reshape(timeseries, (128, 64, n_fields, steps)) )
rescaled_timeseries = (r_timeseries .- μ) ./ σ
##
hfile = h5open("steady_data_2_c.hdf5", "w")
hfile["timeseries"] = rescaled_timeseries
hfile["shift"] = μ
hfile["scaling"] = σ
hfile["sigmax"] = sigmax
close(hfile)