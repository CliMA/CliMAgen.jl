using MLDatasets, MLUtils, Images, DataLoaders, Statistics
using CliMADatasets

"""
Helper function that loads MNIST images and returns loaders.
"""
function get_data_mnist(batchsize; size=32, FT=Float32)
    xtrain, _ = MLDatasets.MNIST(:train; Tx=FT)[:]
    xtrain = Images.imresize(xtrain, (size, size))
    xtrain = reshape(xtrain, size, size, 1, :)
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.MNIST(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (size, size))
    xtest = reshape(xtest, size, size, 1, :)
    xtest = @. 2xtest - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads CIFAR10 images and returns loaders.
"""
function get_data_cifar10(batchsize; size=32, FT=Float32)
    xtrain, _ = MLDatasets.CIFAR10(:train; Tx=FT)[:]
    xtrain = Images.imresize(xtrain, (size, size))
    xtrain = reshape(xtrain, size, size, 3, :)
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.CIFAR10(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (size, size))
    xtest = reshape(xtest, size, size, 3, :)
    xtest = @. 2xtest - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads 2d turbulence images and returns loaders.
"""
function get_data_2dturbulence(batchsize; width=(32, 32), stride=(32, 32), FT=Float32)
    xtrain = CliMADatasets.Turbulence2D(:train; resolution=:high, Tx=FT)[:]
    xtrain = tile_array(xtrain, width[1], width[2], stride[1], stride[2])

    # min-max rescaler
    maxtrain = maximum(xtrain, dims=(1, 2, 4))
    mintrain = minimum(xtrain, dims=(1, 2, 4))
    xtrain = @. 2(xtrain - mintrain) / (maxtrain - mintrain) - 1

    xtrain = MLUtils.shuffleobs(xtrain)

    xtest = CliMADatasets.Turbulence2D(:test; resolution=:high, Tx=FT)[:]
    xtest = tile_array(xtest, width[1], width[2], stride[1], stride[2])

    # apply the same rescaler as on training set
    xtest = @. 2(xtest - mintrain) / (maxtrain - mintrain) - 1

    xtrain = Statistics.mean(xtrain, dims=(1,2))
    xtest = Statistics.mean(xtest, dims=(1,2))

    # mu = Statistics.mean(xtrain, dims=4)
    # sigma = Statistics.std(xtrain, dims=4)
    # xtrain = @. (xtrain - mu) / sigma
    # xtest = @. (xtest - mu) / sigma

    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that tiles an array in the first two spatial dimensions.

TODO!: make work generally for any spatial dimenionality.
"""
function tile_array(A::AbstractArray, xwidth::Int, ywidth::Int, xstride::Int, ystride::Int)
    @assert ndims(A) == 4

    # number of tiles in x and y direction
    xsize, ysize = Base.size(A)[1:2]
    nx = floor(Int, (xsize - xwidth) / xstride)
    ny = floor(Int, (ysize - ywidth) / ystride)

    # tile up the array in spatial directions only!
    processed_data = []
    xranges = map(k -> 1+k*xstride:xwidth+k*xstride, 0:nx)
    yranges = map(k -> 1+k*ystride:ywidth+k*ystride, 0:ny)
    for (xr, yr) in Base.Iterators.product(xranges, yranges)
        push!(processed_data, A[xr, yr, :, :])
    end
    return cat(processed_data..., dims=ndims(A))
end
