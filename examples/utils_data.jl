using MLDatasets, MLUtils, Images, DataLoaders, Statistics
using CliMADatasets
using CliMAgen: expand_dims
"""
Helper function that creates uniform images and returns loaders.
"""
function get_data_uniform(batchsize, std, ndata; size=32, FT=Float32)
    train_means = FT.(randn(ndata)*std)
    test_means = FT.(randn(ndata)*std)
    xtrain = zeros(FT, (size, size, 1, ndata)) .+ expand_dims(train_means, 3)

    maxtrain = maximum(xtrain, dims=(1, 2, 4))
    mintrain = minimum(xtrain, dims=(1, 2, 4))
    xtrain = @. 2(xtrain - mintrain) / (maxtrain - mintrain) - 1
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    
    xtest = zeros(FT, (size, size, 1, ndata)) .+ expand_dims(test_means, 3)
    xtest = @. 2(xtest - mintrain) / (maxtrain - mintrain) - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)
    return (; loader_train, loader_test)
end


"""
Helper function that creates Gaussian images and returns loaders.
"""
function get_data_gaussian(batchsize, mean, std, ndata; size=32, FT=Float32)
    xtrain = randn(FT, (size, size, 1, ndata)) .* std .+mean
    xtest = randn(FT, (size, size, 1, ndata)) .* std .+mean

    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads MNIST images and returns loaders.
"""
function get_data_mnist(batchsize; tilesize=32, FT=Float32)
    xtrain, _ = MLDatasets.MNIST(:train; Tx=FT)[:]
    xtrain = Images.imresize(xtrain, (tilesize, tilesize))
    xtrain = reshape(xtrain, tilesize, tilesize, 1, :)
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.MNIST(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (tilesize, tilesize))
    xtest = reshape(xtest, tilesize, tilesize, 1, :)
    xtest = @. 2xtest - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads FashionMNIST images and returns loaders.
"""
function get_data_fashion_mnist(batchsize; tilesize=32, FT=Float32)
    xtrain, _ = MLDatasets.FashionMNIST(:train; Tx=FT)[:]
    xtrain = Images.imresize(xtrain, (tilesize, tilesize))
    xtrain = reshape(xtrain, tilesize, tilesize, 1, :)
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.FashionMNIST(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (tilesize, tilesize))
    xtest = reshape(xtest, tilesize, tilesize, 1, :)
    xtest = @. 2xtest - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads CIFAR10 images and returns loaders.
"""
function get_data_cifar10(batchsize; tilesize=32, FT=Float32)
    xtrain, _ = MLDatasets.CIFAR10(:train; Tx=FT)[:]
    xtrain = reshape(xtrain, tilesize, tilesize, 3, :)
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.CIFAR10(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (tilesize, tilesize))
    xtest = reshape(xtest, tilesize, tilesize, 3, :)
    xtest = @. 2xtest - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads 2d turbulence images and returns loaders.
"""
function get_data_celeba_hq(batchsize; resolution=32, gender=:male, FT=Float32)
    xtrain = CliMADatasets.CelebAHQ(:train; resolution=resolution, gender=gender, Tx=FT)[:]

    # bring data to [-1, 1] range
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest = CliMADatasets.CelebAHQ(:test; resolution=resolution, gender=gender, Tx=FT)[:]

    # bring data to [-1, 1] range
    xtrain = @. 2xtrain - 1
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
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest = CliMADatasets.Turbulence2D(:test; resolution=:high, Tx=FT)[:]
    xtest = tile_array(xtest, width[1], width[2], stride[1], stride[2])

    # apply the same rescaler as on training set
    xtest = @. 2(xtest - mintrain) / (maxtrain - mintrain) - 1

    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

function get_data_2dturbulence_variant(batchsize; width=(32, 32), stride=(32, 32), FT=Float32)
    xtrain = CliMADatasets.Turbulence2D(:train; resolution=:high, Tx=FT)[:]
    xtrain = tile_array(xtrain, width[1], width[2], stride[1], stride[2])

    # fancy rescaler
    x̄ = mean(xtrain, dims=(1, 2))
    maxtrain_mean = maximum(x̄, dims=4)
    mintrain_mean = minimum(x̄, dims=4)
    Δ̄ = maxtrain_mean .- mintrain_mean
    x̄̃ = @. 2(x̄ -  mintrain_mean) / Δ̄ - 1
    
    xp = xtrain .- x̄
    maxtrain_p = maximum(xp, dims=(1, 2, 4))
    mintrain_p = minimum(xp, dims=(1, 2, 4))
    Δp = maxtrain_p .- mintrain_p
    x̃p = @. 2(xp -  mintrain_p) / Δp - 1

    xtrain = x̄̃ .+ x̃p
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest = CliMADatasets.Turbulence2D(:test; resolution=:high, Tx=FT)[:]
    xtest = tile_array(xtest, width[1], width[2], stride[1], stride[2])

    # apply the same rescaler as on training set
    x̄ = mean(xtest, dims=(1, 2))
    xp = xtest .- x̄
    x̄̃ = @. 2(x̄ - mintrain_mean) / Δ̄ - 1
    x̃p = @. 2(xp - mintrain_p) / Δp - 1

    xtest = x̄̃ .+ x̃p
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that tiles an array in the first two spatial dimensions.

Tiles wrap around periodically if input width is larger than spatial size of array.
TODO!: make work generally for any spatial dimenionality.
"""
function tile_array(A::AbstractArray, xwidth::Int, ywidth::Int, xstride::Int, ystride::Int)
    @assert ndims(A) == 4

    # number of tiles in x and y direction
    xsize, ysize = Base.size(A)[1:2]
    nx = floor(Int, abs(xsize - xwidth) / xstride)
    ny = floor(Int, abs(ysize - ywidth) / ystride)

    # tile up the array in spatial directions only!
    processed_data = []
    xranges = map(k -> 1+k*xstride:xwidth+k*xstride, 0:nx)
    yranges = map(k -> 1+k*ystride:ywidth+k*ystride, 0:ny)

    # if width > size of array, we wrap around periodically
    xranges = map(x -> map(y -> mod(y, xsize) != 0 ? mod(y, xsize) : xsize, x), xranges)
    yranges = map(x -> map(y -> mod(y, ysize) != 0 ? mod(y, ysize) : ysize, x), yranges)
    for (xr, yr) in Base.Iterators.product(xranges, yranges)
        push!(processed_data, A[xr, yr, :, :])
    end
    return cat(processed_data..., dims=ndims(A))
end
