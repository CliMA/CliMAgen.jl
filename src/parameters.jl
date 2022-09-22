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

"""
    CliMAgen.parse_commandline
"""
function parse_commandline()
    s = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! s begin
        "--project"
        help = "Wandb project name"
        arg_type = String
        default = "CliMAgen.jl"
        required = false
        "--restartfile"
        help = "Restart from this checkpoint file"
        arg_type = String
        default = nothing
        required = false
        "--savedir"
        help = "Output directory for checkpoint files and artifacts"
        arg_type = String
        default = "./output/"
        required = false
        "--seed"
        help = "Random seed"
        arg_type = Int
        default = 123
        required = false
        "--logging"
        help = "Toggle logging"
        action = :store_true
        "--nogpu"
        help = "Toggle GPU usage"
        action = :store_true
    end

    return ArgParse.parse_args(s) # returns a dictionary
end
