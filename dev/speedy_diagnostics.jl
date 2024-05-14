struct MyInterpolatedField{Grid, NF_model, Grid_model, FieldName} <: SpeedyWeather.AbstractCallback
    "Interpolate only when scheduled"
    schedule::Schedule

    "From which layer to interpolate"
    layer::Int

    "Variable on interpolated grid"
    var::Grid

    "Interpolator to interpolate model grid into var"
    interpolator::RingGrids.AnvilInterpolator{NF_model, Grid_model}

    "Symbol for Field"
    field_name::FieldName
end

function MyInterpolatedField(
    SG::SpectralGrid;
    layer::Integer = 1,
    nlat_half::Integer = 32,
    NF::Type{<:AbstractFloat} = Float32,
    Grid::Type{<:RingGrids.AbstractGrid} = FullGaussianGrid{NF},
    schedule::Schedule = Schedule(every=Day(1)),
    field_name::Symbol = :temp_grid
)
    n_points = RingGrids.get_npoints(Grid, nlat_half)
    var = zeros(Grid, nlat_half) 
    interpolator = RingGrids.AnvilInterpolator(SG.NF, SG.Grid, SG.nlat_half, n_points)
    RingGrids.update_locator!(interpolator, RingGrids.get_latdlonds(var)...)
    MyInterpolatedField{Grid, SG.NF, SG.Grid, Symbol}(schedule, layer, var, interpolator, field_name)
end

function SpeedyWeather.initialize!(
    callback::MyInterpolatedField,
    progn::PrognosticVariables,
    diagn::DiagnosticVariables,
    model::ModelSetup,
)
    initialize!(callback.schedule, progn.clock)
end

function SpeedyWeather.callback!(
    callback::MyInterpolatedField,
    progn::PrognosticVariables,
    diagn::DiagnosticVariables,
    model::ModelSetup,
)
    isscheduled(callback.schedule, progn.clock) || return nothing
    k = callback.layer
    field =getfield(diagn.layers[k].grid_variables, callback.field_name)
    (;var, interpolator) = callback
    RingGrids.interpolate!(var, field, interpolator)
end

SpeedyWeather.finish!(::MyInterpolatedField, args...) = nothing

##

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