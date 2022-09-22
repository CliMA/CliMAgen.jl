"""
Helper function to convert a parameters struct to a dict recursively.

The idea is that parameter structs only contain reals, Booleans, or strings
and hence can be easily unpacked into dictionaries.
"""
struct2dict(s::Union{Real,String,Bool}) = s
struct2dict(s) = Dict(x => struct2dict(getfield(s, x)) for x in fieldnames(typeof(s)))
dict2nt(d::Dict) = NamedTuple{Tuple(Symbol(k) for k in keys(d))}((values(d)...,))

"""
    ClimaGen.save_model_and_optimizer
"""
function save_model_and_optimizer(model, opt, hparams::HyperParameters, path::String)
    BSON.@save path model opt hparams
    @info "Model saved at $(path)."
end

"""
    ClimaGen.load_model_and_optimizer
"""
function load_model_and_optimizer(path::String)
    BSON.@load path model opt hparams
    return (; model=model, opt=opt, hparams=hparams)
end
