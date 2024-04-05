struct MyInterpolatedTemperature{Grid, NF_model, Grid_model} <: SpeedyWeather.AbstractCallback
    "Interpolate only when scheduled"
    schedule::Schedule

    "From which layer to interpolate"
    layer::Int

    "Variable on interpolated grid"
    var::Grid

    "Interpolator to interpolate model grid into var"
    interpolator::RingGrids.AnvilInterpolator{NF_model, Grid_model}
end

function MyInterpolatedTemperature(
    SG::SpectralGrid;
    layer::Integer = 1,
    nlat_half::Integer = 32,
    NF::Type{<:AbstractFloat} = Float32,
    Grid::Type{<:RingGrids.AbstractGrid} = FullGaussianGrid{NF},
    schedule::Schedule = Schedule(every=Day(1))
)
    n_points = RingGrids.get_npoints(Grid, nlat_half)
    var = zeros(Grid, nlat_half) 
    interpolator = RingGrids.AnvilInterpolator(SG.NF, SG.Grid, SG.nlat_half, n_points)
    RingGrids.update_locator!(interpolator, RingGrids.get_latdlonds(var)...)
    MyInterpolatedTemperature{Grid, SG.NF, SG.Grid}(schedule, layer, var, interpolator)
end

function SpeedyWeather.initialize!(
    callback::MyInterpolatedTemperature,
    progn::PrognosticVariables,
    diagn::DiagnosticVariables,
    model::ModelSetup,
)
    initialize!(callback.schedule, progn.clock)
end

function SpeedyWeather.callback!(
    callback::MyInterpolatedTemperature,
    progn::PrognosticVariables,
    diagn::DiagnosticVariables,
    model::ModelSetup,
)
    isscheduled(callback.schedule, progn.clock) || return nothing
    k = callback.layer
    (;temp_grid) = diagn.layers[k].grid_variables
    (;var, interpolator) = callback
    RingGrids.interpolate!(var, temp_grid, interpolator)
end

SpeedyWeather.finish!(::MyInterpolatedTemperature, args...) = nothing