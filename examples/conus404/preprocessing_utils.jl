using FFTW
using Statistics
using CliMAgen: AbstractPreprocessing
import CliMAgen: apply_preprocessing, invert_preprocessing

struct Conus404Preprocessing{FT, A} <: AbstractPreprocessing{FT}
    low_pass::Bool
    low_pass_k::Union{Nothing, Int}
    mean_map::A
    std_map::A
end

function Conus404Preprocessing{FT}(xtrain; low_pass = false, low_pass_k = nothing) where {FT}
    if low_pass
        xtrain = lowpass_filter(xtrain, low_pass_k)
    end
    # TODO - do we want this per pixel? or averaged over space?
    # TODO - for precip, we want to use the mean and variance of the nonzero pixels
    # TODO - instead of setting the preprocessed data to have sigma =1, do we want more like ~0.5 to keep data more between -1 and 1?
    temporal_mean = mean(xtrain, dims = 4)
    temporal_std = std(xtrain, dims = 4)
    return Conus404Preprocessing{FT, typeof(temporal_mean)}(low_pass, low_pass_k, temporal_mean, temporal_std)
end

function apply_preprocessing(x, scaling::Conus404Preprocessing)
    x̃ = similar(x)
    x̃ .= @. (x - scaling.mean_map) / scaling.std_map
    return x̃
end

function invert_preprocessing(x̃, scaling::Conus404Preprocessing)
    x = similar(x̃)
    x .= @. x̃ * scaling.std_map + scaling.mean_map
    return x
end

function save_preprocessing_params(
    x, preprocess_params_file;
    standard_scaling=true,
    low_pass=false,
    low_pass_k=nothing,
    FT=Float32,
)

    if standard_scaling
        #scale means and spatial variations separately
        x̄ = mean(x, dims=(1, 2))
        max_mean = maximum(x̄, dims=4)
        min_mean = minimum(x̄, dims=4)
        Δ̄ = max_mean .- min_mean
        xp = x .- x̄
        max_p = maximum(xp, dims=(1, 2, 4))
        min_p = minimum(xp, dims=(1, 2, 4))
        Δp = max_p .- min_p
        
        # To prevent dividing by zero
        Δ̄[Δ̄ .== 0] .= FT(1)
        Δp[Δp .== 0] .= FT(1)
        scaling = MeanSpatialScaling{FT}(min_mean, Δ̄, min_p, Δp)
    else
        scaling = Conus404Preprocessing{FT}(x; low_pass=low_pass, low_pass_k=low_pass_k)
    end
    JLD2.save_object(preprocess_params_file, scaling)
end
    
"""
    lowpass_filter(x, k)

Lowpass filters the data `x`, assumed to be structured as
dxdxCxB, where d is the number of spatial pixels, C is the number
of channels, and B is the number of batch members, such that
all power above wavenumber `k` = kx = ky is set to zero.
"""
function lowpass_filter(x, k)
    d = size(x)[1]
    if iseven(k)
        k_ny = Int64(k/2+1)
    else
        k_ny = Int64((k+1)/2)
    end
    FT = eltype(x)
    y = Complex{FT}.(x)
    fft!(y, (1,2));
    # Filter. The indexing here is specific to how `fft!` stores the 
    # Fourier transformed image.
    y[:,k_ny:(d-k_ny),:,:] .= Complex{FT}(0);
    y[k_ny:(d-k_ny),:,:,:] .= Complex{FT}(0);
    ifft!(y, (1,2))
    return real(y)
end
