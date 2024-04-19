# 
## simulation.prognostic_variables.surface.timesteps[1].pres


struct MyInterpolatedPressure{Grid, NF_model, Grid_model, FieldName} <: SpeedyWeather.AbstractCallback
    "Interpolate only when scheduled"
    schedule::Schedule

    "Variable on interpolated grid"
    var::Grid

    "Interpolator to interpolate model grid into var"
    interpolator::RingGrids.AnvilInterpolator{NF_model, Grid_model}
end

function MyInterpolatedPressure(
    SG::SpectralGrid;
    nlat_half::Integer = 32,
    NF::Type{<:AbstractFloat} = Float32,
    Grid::Type{<:RingGrids.AbstractGrid} = FullGaussianGrid{NF},
    schedule::Schedule = Schedule(every=Day(1)),
)
    n_points = RingGrids.get_npoints(Grid, nlat_half)
    var = zeros(Grid, nlat_half) 
    interpolator = RingGrids.AnvilInterpolator(SG.NF, SG.Grid, SG.nlat_half, n_points)
    RingGrids.update_locator!(interpolator, RingGrids.get_latdlonds(var)...)
    MyInterpolatedPressure{Grid, SG.NF, SG.Grid, Symbol}(schedule, var, interpolator)
end

function SpeedyWeather.initialize!(
    callback::MyInterpolatedPressure,
    progn::PrognosticVariables,
    diagn::DiagnosticVariables,
    model::ModelSetup,
)
    initialize!(callback.schedule, progn.clock)
end

function SpeedyWeather.callback!(
    callback::MyInterpolatedPressure,
    progn::PrognosticVariables,
    diagn::DiagnosticVariables,
    model::ModelSetup,
)
    isscheduled(callback.schedule, progn.clock) || return nothing
    field = diagn.surface.pres_grid
    (; var, interpolator) = callback
    RingGrids.interpolate!(var, field, interpolator)
end

SpeedyWeather.finish!(::MyInterpolatedPressure, args...) = nothing