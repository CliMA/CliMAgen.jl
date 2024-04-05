using SpeedyWeather
using CliMAgen
using Flux
using TOML
using BSON
using ProgressBars
include("./online_utils.jl")
include("./stochastic_stirring_def.jl")

FT = Float32
experiment_toml = "./speedy_weather.toml"
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
!ispath(params.experiment.savedir) && mkpath(params.experiment.savedir)
model, model_smooth, opt, opt_smooth, ps, ps_smooth, lossnf = setup_model(params;FT=FT)
freq_chckpt = params.training.freq_chckpt

#=
spectral_grid = SpectralGrid(trunc=31, Grid=OctahedralGaussianGrid, nlev=8)
model = BarotropicModel() # PrimitiveDryModel(;spectral_grid, orography = EarthOrography(spectral_grid))
=#
#=



=#
simulation = initialize!(model)
run!(simulation,period=Day(10000),output=true)
# https://github.com/SpeedyWeather/SpeedyWeather.jl/blob/main/src/dynamics/forcing.jl#L47C1-L47C17
u = simulation.model.output.u


