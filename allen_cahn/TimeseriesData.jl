using HDF5, DataLoaders, Random
import DataLoaders.getobs, DataLoaders.nobs
using CliMAgen
import CliMAgen: GaussianFourierProjection
using Distributions

function GaussianFourierProjection(embed_dim::Int, embed_dim2::Int, scale::FT) where {FT}
    Random.seed!(1234) # same thing every time
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

##
function DataLoaders.getobs(d::TimeseriesData, i::Int)
    data_size = size(d.data)
    N = length(data_size)
    rdata = reshape(d.data, (32, 32, 128, 2000))
    #=
    index_lags = [collect(0:30)..., 1000] # collect(0:1000)
    shift = rand(index_lags)
    =#
    index_lags = 0:1000
    ps = @. exp(-index_lags/30)
    ps = ps / sum(ps)
    tmp = DiscreteNonParametric(index_lags, ps)
    shift = rand(tmp)
    ensemble_member = rand(1:128)
    i = rand(1:2000)
    j = ((i + shift) > 2000) ? 2000 : (i + shift)
    data1 = rdata[:, :, ensemble_member, i]
    data2 = rdata[:, :, ensemble_member, j]
    τ = (j-i) / index_lags[end]
    T = gfp(τ) 
    return cat(data1, data2, T; dims = 3)
end

function DataLoaders.nobs(d::TimeseriesData)
    data_size = size(d.data)
    return data_size[end]
end