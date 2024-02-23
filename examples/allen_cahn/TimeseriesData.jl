using HDF5, DataLoaders
import DataLoaders.getobs, DataLoaders.nobs

function time_embedding(τ, data_size)
    N = data_size[1]
    M = data_size[2]
    xs = reshape(collect((0:N-1)/N), (N, 1))
    ys = reshape(collect((0:M-1)/M), (1, M))
    return sin.((xs .- τ/2) * 2π) .* sin.((ys .- τ/2) * 2π)
end

struct TimeseriesData{S}
    data::S
end
function DataLoaders.getobs(d::TimeseriesData, i::Int)
    data_size = size(d.data)
    N = length(data_size)
    j = rand(i:data_size[end])
    τ = (j-i) / data_size[end]
    T = time_embedding(τ, data_size)
    return cat(d.data[[(:) for _ in 1:N-1]..., i], d.data[[(:) for _ in 1:N-1]..., j], T; dims = 3)
end
function DataLoaders.nobs(d::TimeseriesData)
    data_size = size(d.data)
    return data_size[end]
end