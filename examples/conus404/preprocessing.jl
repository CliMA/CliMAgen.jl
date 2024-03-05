using FFTW
using Statistics
using CliMAgen: AbstractPreprocessing
import CliMAgen: apply_preprocessing, invert_preprocessing

struct Conus404Preprocessing{FT, A} <: AbstractPreprocessing{FT}
    low_pass::Bool
    low_pass_k::Union{Nothing, Int}
    temperature_mean_map::A
    temperature_std_map::A
end

function Conus404Preprocessing{FT}(xtrain; low_pass = false, low_pass_k = nothing) where {FT}
    if low_pass
        xtrain = lowpass_filter(xtrain, low_pass_k)
    end
    temporal_mean = mean(xtrain, dims = 4)
    temporal_std = std(xtrain, dims = 4)
    return Conus404Preprocessing{FT, typeof(temporal_mean)}(low_pass, low_pass_k, temporal_mean, temporal_std)
end

function apply_preprocessing(x, scaling::Conus404Preprocessing)
    temperature_channel = 1
    x̃ = similar(x)
    x̃[:,:,temperature_channel,:] .= @. (x[:,:,temperature_channel,:] - scaling.temperature_mean_map)/scaling.temperature_std_map
    return x̃
end


function invert_preprocessing(x̃, scaling::Conus404Preprocessing)
    temperature_channel = 1
    x = similar(x̃)
    x[:,:,temperature_channel,:] .= @. x̃[:,:,temperature_channel,:] * scaling.temperature_std_map  + scaling.temperature_mean_map
    return x
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
