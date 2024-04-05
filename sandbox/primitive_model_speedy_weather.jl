using SpeedyWeather
include("my_field.jl")
# components
spectral_grid = SpectralGrid(trunc=40, nlev=8) # SpectralGrid(trunc=31, nlev=5)
ocean = AquaPlanet(spectral_grid, temp_equator=302, temp_poles=273)
land_sea_mask = AquaPlanetMask(spectral_grid)
orography = NoOrography(spectral_grid)
model = PrimitiveWetModel(; spectral_grid, ocean) # , land_sea_mask, orography)
simulation = initialize!(model)
# :nlat_half, :vor_grid, :div_grid,
#  :temp_grid, :temp_grid_prev, :temp_virt_grid, :humid_grid, 
# :u_grid, :v_grid, :u_grid_prev, :v_grid_prev
my_fields = []
for field in [:temp_grid, :vor_grid, :humid_grid, :div_grid]
    my_field_on_1 = MyInterpolatedField(spectral_grid; schedule = Schedule(every=Day(1)), field_name = field)
    my_field = deepcopy(my_field_on_1)
    push!(my_fields, my_field)
    add!(model.callbacks, my_field)
end
run!(simulation, period=Day(50))

using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1]; title = "Temperature")
lat, lon = RingGrids.get_latdlonds(my_fields[1].var)
heatmap!(ax, reshape(lon, (128, 64))[:,1], reshape(lat, (128, 64))[1, :], reshape(my_fields[1].var, (128, 64)))
ax = Axis(fig[1, 2]; title = "Vorticity")
heatmap!(ax, reshape(lon, (128, 64))[:,1], reshape(lat, (128, 64))[1, :], reshape(my_fields[2].var, (128, 64)))
ax = Axis(fig[2, 1]; title = "Humidity")
heatmap!(ax, reshape(lon, (128, 64))[:,1], reshape(lat, (128, 64))[1, :], reshape(my_fields[3].var, (128, 64)))
ax = Axis(fig[2, 2]; title = "Divergence")
heatmap!(ax, reshape(lon, (128, 64))[:,1], reshape(lat, (128, 64))[1, :], reshape(my_fields[4].var, (128, 64)))
save("fields.png", fig)