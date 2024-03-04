using HDF5, DataLoaders
import DataLoaders.getobs, DataLoaders.nobs
using CliMAgen
import CliMAgen: GaussianFourierProjection

function GaussianFourierProjection(embed_dim::Int, embed_dim2::Int, scale::FT) where {FT}
    W = randn(FT, embed_dim ÷ 2, embed_dim2) .* scale
    return GaussianFourierProjection(W)
end

gfp = GaussianFourierProjection(32, 32, 30.0f0)

function time_embedding(τ, data_size)
    N = data_size[1]
    M = data_size[2]
    xs = reshape(collect((0:N-1)/N), (N, 1))
    ys = reshape(collect((0:M-1)/M), (1, M))
    return sin.((xs .- τ/4) * 2π) .* sin.((ys .- τ/4) * 2π) 
end

struct TimeseriesData{S}
    data::S
end
function DataLoaders.getobs(d::TimeseriesData, i::Int)
    data_size = size(d.data)
    N = length(data_size)
    minval = 1
    skip = 10
    # j = rand(i:data_size[end])
    # τ = (j-i) / data_size[end]
    shift = rand(0:1)
    j = ( (i + skip * shift * 128) > data_size[end] ) ? i : (i + skip * shift * 128)  # 128 comes from ensemble member
    if j == i
        shift = 0
    end
    τ = shift / minval
    T = gfp(τ) # time_embedding(τ, data_size)
    return cat(d.data[[(:) for _ in 1:N-1]..., i], d.data[[(:) for _ in 1:N-1]..., j], T; dims = 3)
end
function DataLoaders.nobs(d::TimeseriesData)
    data_size = size(d.data)
    return data_size[end]
end