"""
    AbstractPreprocessing{FT<:AbstractFloat}

An abstract type for all preprocessing transformations.
"""
abstract type AbstractPreprocessing{FT<:AbstractFloat} end

"""
    StandardScaling{FT} <: AbstractPreprocessing{FT}

A struct that holds the minimum and maximum by channel, over pixels and 
over all samples, for the training data.
"""
struct StandardScaling{FT} <: AbstractPreprocessing{FT}
    mintrain::Array{FT}
    maxtrain::Array{FT}
end

"""
MeanSpatialScaling{FT} <: AbstractPreprocessing{FT}

A struct that holds the minimum and range 
both by spatial and mean component, and by channel, over pixels and 
over all samples, for the training data.
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
function invert_preprocessing(x̃, scaling::MeanSpatialScaling)
    (; mintrain_mean, Δ̄, mintrain_p, Δp) = scaling
    Δ̄ .= prevent_divide_by_zero.(Δ̄)
    Δp .= prevent_divide_by_zero.(Δp)

    tmp = @. (x̃ + 2) / 2 * Δp + mintrain_p
    xp = tmp .- Statistics.mean(tmp, dims = (1,2))
    x̄ = @. (tmp - xp) / Δp * Δ̄ + mintrain_mean
    return xp .+ x̄
end

"""
    invert_preprocessing(x̃, scaling::StandardScaling)

Computes the inverse preprocess transform of the data, given a
scaling of type `StandardScaling`.
"""
function invert_preprocessing(x̃, scaling::StandardScaling)
    (; maxtrain, mintrain) = scaling
    Δ = maxtrain - mintrain
    Δ .= prevent_divide_by_zero.(Δ)
    return @. (x̃ + 1) / 2 * Δ + mintrain
end

"""
    apply_preprocessing(x, scaling::StandardScaling)

Computes the preprocessing transform of the data x, given a
scaling of type `StandardScaling`.
"""
function apply_preprocessing(x, scaling::StandardScaling)
    (; maxtrain, mintrain) = scaling
    Δ = maxtrain - mintrain
    Δ .= prevent_divide_by_zero.(Δ)
    return @. 2 * (x - mintrain) / Δ - 1
end

"""
    apply_preprocessing(x, scaling::MeanSpatialScaling)

Computes the preprocessing transform of the data x, given a
scaling of type `MeanSpatialScaling`.
"""
function apply_preprocessing(x, scaling::MeanSpatialScaling)
    (; mintrain_mean, Δ̄, mintrain_p, Δp) = scaling
    Δ̄ .= prevent_divide_by_zero.(Δ̄)
    Δp .= prevent_divide_by_zero.(Δp)

    x̄ = Statistics.mean(x, dims=(1, 2))
    xp = x .- x̄
    x̄̃ = @. 2(x̄ -  mintrain_mean) / Δ̄ - 1
    x̃p = @. 2(xp -  mintrain_p) / Δp - 1
    return  x̄̃ .+ x̃p
end

prevent_divide_by_zero(x::FT) where{FT} = max(abs(x),eps(FT))