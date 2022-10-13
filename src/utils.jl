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
dict2nt(s::Union{Real,String,Bool}) = s
dict2nt(d::Dict) = NamedTuple{Tuple(Symbol(k) for k in keys(d))}((dict2nt.(values(d))...,))
