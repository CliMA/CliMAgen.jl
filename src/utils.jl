"""
Helper functions that adds `dims` dimensions to the front of a `AbstractVecOrMat`.
Similar in spirit to TensorFlow's `expand_dims` function.

# References:
https://www.tensorflow.org/api_docs/python/tf/expand_dims
"""
expand_dims(x::Real, kwargs...) = x
expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))

"""
Helper functions to convert a parameters struct to a dict recursively.

The idea is that parameter structs only contain reals, Booleans, or strings
and hence can be easily unpacked into dictionaries.
"""
struct2dict(s::Union{Real,String,Bool}) = s
struct2dict(s) = Dict(x => struct2dict(getfield(s, x)) for x in fieldnames(typeof(s)))
dict2nt(d::Dict) = NamedTuple{Tuple(Symbol(k) for k in keys(d))}((values(d)...,))

"""
Helper function that parses commandline.
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
