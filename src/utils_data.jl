"""
Helper function that loads MNIST images and returns loaders.
"""
function get_data_mnist(hptrain; size=32, FT=Float32)
    batchsize, = hptrain

    xtrain, _ = MLDatasets.MNIST(:train; Tx=FT)[:]
    xtrain = Images.imresize(xtrain, (size, size))
    xtrain = reshape(xtrain, size, size, 1, :)
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.MNIST(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (size, size))
    xtest = reshape(xtest, size, size, 1, :)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads CIFAR10 images and returns loaders.

Note: Song etal. do a "random flip".
https://github.com/yang-song/score_sde/blob/main/configs/default_cifar10_configs.py#L44
"""
function get_data_cifar10(hptrain; size=32, FT=Float32)
    batchsize, = hptrain

    xtrain, _ = MLDatasets.CIFAR10(:train; Tx=FT)[:]
    xtrain = Images.imresize(xtrain, (size, size))
    xtrain = reshape(xtrain, size, size, 3, :)
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.CIFAR10(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (size, size))
    xtest = reshape(xtest, size, size, 3, :)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end
