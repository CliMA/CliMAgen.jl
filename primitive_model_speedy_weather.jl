using SpeedyWeather

# components
spectral_grid = SpectralGrid(trunc=63, nlev=8) # SpectralGrid(trunc=31, nlev=5)
ocean = AquaPlanet(spectral_grid, temp_equator=302, temp_poles=273)
land_sea_mask = AquaPlanetMask(spectral_grid)
orography = NoOrography(spectral_grid)

# create model, initialize, run
model = PrimitiveWetModel(; spectral_grid, ocean) # , land_sea_mask, orography)
simulation = initialize!(model)
run!(simulation, period=Day(50))
