abstract type AbstractParams end

"""
    ClimaGen.HyperParameters
"""
Base.@kwdef struct HyperParameters <: AbstractParams
    data::NamedTuple
    model::NamedTuple
    optimizer::NamedTuple
    training::NamedTuple
end
