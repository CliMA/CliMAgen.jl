include("YeoJohnsonTransform.jl")
using .YeoJohnsonTransform
"""
    AbstractPreprocessing{FT<:AbstractFloat}

An abstract type for preprocessing transformations.
"""
abstract type AbstractPreprocessing{FT<:AbstractFloat} end


"""
    PowerTransform{FT} <: AbstractPreprocessing{FT}

A struct that holds the parameters needed for the 
Yeo-Johnson power transform.
"""
struct PowerTransform{FT} <: AbstractPreprocessing{FT}
    λ::Array{FT}
end

function PowerTransform(x)
    FT = eltype(x)
    datasize = size(x)
    ndims = length(datasize)
    npixels = prod(datasize[1:nspatial])
    pixeldims = 1:nspatial
    nchannels = datasize[end-1]
    channeldim = ndims-1
    nsamples = datasize[end]
    sampledim = ndims
    nrequired = min(nsamples*npixels, 50000)
    perm = [channeldim, pixeldims..., sampledim]
    y = reshape(permutedims(x, perm), (nchannels, npixels*nsamples))[:,1:nrequired];
    output = mapslices(YeoJohnsonTransform.lambda, y;dims = 2)
    λ = [o.value for o in output][:]
    return PowerTransform{FT}(λ)
end


"""
    apply_preprocessing(x, scaling::PowerTransform)

Preprocesses the data x, given a scaling of type `PowerTransform`.
"""
function apply_preprocessing(x, scaling::PowerTransform;FT = Float32)
    λ = scaling.λ
    datasize = size(x)
    ndims = length(datasize)
    channeldim = ndims -1
    x̃ = similar(x)
    wrapper(x, l) = YeoJohnsonTransform.transform.(x,l)
    eachslice(x̃, dims = channeldim) .= wrapper.(eachslice(x, dims = channeldim),λ)
    return x̃ 
end


"""
    apply_preprocessing(x, scaling::PowerTransform)

inverts the preprocessing of the data x, given a scaling of type `PowerTransform`.
"""
function invert_preprocessing(x̃, scaling::PowerTransform;FT = Float32)
    λ = scaling.λ
    datasize = size(x̃)
    ndims = length(datasize)
    channeldim = ndims -1
    x = similar(x̃)
    wrapper(x̃, l) = YeoJohnsonTransform.inversetransform.(x̃,l)
    eachslice(x, dims = channeldim) .= wrapper.(eachslice(x̃, dims = channeldim),λ)
    return x
end



"""
    StandardScaling{FT} <: AbstractPreprocessing{FT}

A struct that holds the minimum and range by channel, 
where the minimum and range are taken over pixels and 
over all samples of the training data.
"""
struct StandardScaling{FT} <: AbstractPreprocessing{FT}
    mintrain::Array{FT}
    Δ::Array{FT}
end

"""
    MeanSpatialScaling{FT} <: AbstractPreprocessing{FT}

A struct that holds the minimum and range by spatial 
(denoted `p`) and mean component (denoted with overbars),
 and by channel, where the minimum and range are taken over 
pixels and over all samples of the training data.
"""
struct MeanSpatialScaling{FT} <: AbstractPreprocessing{FT}
    mintrain_mean::Array{FT}
    Δ̄::Array{FT}
    mintrain_p::Array{FT}
    Δp::Array{FT}
end

"""
    invert_preprocessing(x̃, scaling::MeanSpatialScaling)

Computes the inverse preprocess transform of the data, given a
scaling of type `MeanSpatialScaling`.
"""
function invert_preprocessing(x̃, scaling::MeanSpatialScaling; nspatial=2)
    (; mintrain_mean, Δ̄, mintrain_p, Δp) = scaling
    tmp = @. (x̃ + 2) / 2 * Δp + mintrain_p
    xp = tmp .- Statistics.mean(tmp, dims = (1:nspatial))
    x̄ = @. (tmp - xp) / Δp * Δ̄ + mintrain_mean
    return xp .+ x̄
end

"""
    invert_preprocessing(x̃, scaling::StandardScaling)

Computes the inverse preprocess transform of the data, given a
scaling of type `StandardScaling`.
"""
function invert_preprocessing(x̃, scaling::StandardScaling; nspatial=2)
    (; mintrain, Δ) = scaling
    return @. (x̃ + 1) / 2 * Δ + mintrain
end

"""
    apply_preprocessing(x, scaling::StandardScaling)

Preprocesses the data x, given a scaling of type `StandardScaling`.
"""
function apply_preprocessing(x, scaling::StandardScaling; nspatial=2)
    (; mintrain, Δ) = scaling
    return @. 2 * (x - mintrain) / Δ - 1
end

"""
    apply_preprocessing(x, scaling::MeanSpatialScaling)


Preprocesses the data x, given a scaling of type `MeanSpatialScaling`.
"""
function apply_preprocessing(x, scaling::MeanSpatialScaling; nspatial=2)
    (; mintrain_mean, Δ̄, mintrain_p, Δp) = scaling
    x̄ = Statistics.mean(x, dims=(1:nspatial))
    xp = x .- x̄
    x̄̃ = @. 2(x̄ -  mintrain_mean) / Δ̄ - 1
    x̃p = @. 2(xp -  mintrain_p) / Δp - 1
    return  x̄̃ .+ x̃p
end
