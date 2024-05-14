using SpeedyWeather
using Distributions    # just for Uniform(a,b)
using Random
# Random.seed!(12345)

include("speedy_diagnostics.jl")

add_pressure_field = false
fields =  [:temp_grid] # [:temp_grid, :vor_grid, :humid_grid, :div_grid]
layers = [5] #  [5, 4, 3, 2, 1] # bottom to top

function generate_parameters(; default = false)
    if default
        default_parameters = (; 
        rotation = 7.29e-5, 

        relax_time_slow = 40*Day(1),
        relax_time_fast = 4*Day(1),
        Tmin = 200,
        Tmax = 315,
        ΔTy = 60,
        Δθz = 10,

        time_scale = 4*Hour(1),
        relative_humidity = 0.7,

        f_wind = 0.95,
        V_gust = 5,

        orography_scale = 1
        )
        return default_parameters
    else
        parameters = (;
        rotation = rand(Uniform(-1e-4, 1e-4)),

        relax_time_slow = Second(round(Int, 24*3600*rand(Uniform(20,80)))),
        relax_time_fast = Second(round(Int, 24*3600*rand(Uniform(2,8)))),
        Tmin = rand(Uniform(190,210)),
        Tmax = rand(Uniform(305,325)),
        ΔTy = rand(Uniform(50,70)),
        Δθz = rand(Uniform(5,15)),

        time_scale = Second(round(Int, 3600*rand(Uniform(2,16)))),
        relative_humidity = rand(Uniform(0.6,0.8)),

        f_wind = rand(Uniform(0,1)),
        V_gust = rand(Uniform(0,10)),

        orography_scale = rand(Uniform(0,1))
        )
    end
    return parameters
end

function speedy_sim(; parameters, layers, fields, add_pressure_field)
    @info "Building Simulation"
    spectral_grid = SpectralGrid(trunc=31, nlev=5)

    model = PrimitiveWetModel(;
        spectral_grid,

        planet = Earth(spectral_grid,
            rotation=parameters.rotation, # default 7.29e-5 1/s
        ),

        # Held-Suarez forcing but not its drag
        temperature_relaxation = HeldSuarez(spectral_grid,
            relax_time_slow=parameters.relax_time_slow,  # default 40 days
            relax_time_fast=parameters.relax_time_fast,  # default 4 days
            Tmin=parameters.Tmin,           # default 200K
            Tmax=parameters.Tmax,           # default 315K
            ΔTy=parameters.ΔTy,             # default 60K
            Δθz=parameters.Δθz,             # default 10K
        ),

        # other physics
        convection = SimplifiedBettsMiller(spectral_grid,
            time_scale= parameters.time_scale,                # default 4 hours
            relative_humidity=parameters.relative_humidity,   # default 0.7
        ),
        large_scale_condensation = ImplicitCondensation(spectral_grid),
        shortwave_radiation = NoShortwave(),
        longwave_radiation = NoLongwave(),
        vertical_diffusion = NoVerticalDiffusion(),
        # vertical_diffusion = BulkRichardsonDiffusion(spectral_grid), # maybe needed

        # Surface fluxes
        boundary_layer_drag = BulkRichardsonDrag(spectral_grid),
        surface_wind = SurfaceWind(spectral_grid,
            f_wind = parameters.f_wind,          # surface wind scale, default 0.95
            V_gust = parameters.V_gust,         # wind gusts, default 5m/s
        ),

        # use Earth's orography
        orography = EarthOrography(spectral_grid),
    )

    my_fields = []
    for layer in layers, field in fields
        my_field = MyInterpolatedField(spectral_grid; schedule = Schedule(every=Day(1)), field_name = field, layer = layer)
        push!(my_fields, my_field)
        add!(model.callbacks, my_field)
    end
    if add_pressure_field
        my_field = MyInterpolatedPressure(spectral_grid; schedule = Schedule(every=Day(1)))
        push!(my_fields, my_field)
        add!(model.callbacks, my_field)
    end

    simulation = initialize!(model)

    # scale orography manually
    orography_scale = parameters.orography_scale
    model.orography.orography .*= orography_scale
    model.orography.geopot_surf .*= orography_scale
    return simulation, my_fields
end

# Check if the simulation runs 
parameters = generate_parameters(default=true)
simulation, my_fields = speedy_sim(; parameters, layers, fields, add_pressure_field)
run!(simulation, period=Day(100))
plot(my_fields[1].var)