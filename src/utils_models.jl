"""
Helper function that adds `dims` dimensions to the front of a `AbstractVecOrMat`.
Similar in spirit to TensorFlow's `expand_dims` function.

# References:
https://www.tensorflow.org/api_docs/python/tf/expand_dims
"""
expand_dims(x::Real, kwargs...) = x
expand_dims(x::AbstractVecOrMat, dims::Int=2) = reshape(x, (ntuple(i -> 1, dims)..., size(x)...))