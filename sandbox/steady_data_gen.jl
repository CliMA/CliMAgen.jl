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
steps = 10
timeseries = zeros(spectral_grid.NF, my_fields[1].interpolator.locator.npoints, length(my_fields), steps)
for j in ProgressBar(1:steps)
    run!(simulation, period=Day(1))
    for (i, my_field) in enumerate(my_fields)
        timeseries[:, i, j] = copy(my_fields[i].var)
    end
end

n_fields = length(my_fields)
r_timeseries = reshape(timeseries, (128, 64, n_fields, steps)) 
μ = mean(timeseries, dims = (1, 2, 4))
σ = reshape([quantile(abs.(r_timeseries .- μ)[:], 0.99) for i in 1:n_fields], (1, 1, n_fields, 1))

rescaled_timeseries = (r_timeseries .- μ) ./ σ