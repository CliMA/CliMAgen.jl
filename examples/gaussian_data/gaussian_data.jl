using CliMAgen: StandardScaling, apply_preprocessing
using JLD2
using MLUtils
using DataLoaders

function get_data_gaussian(batchsize, preprocess_params_file; tilesize=16, FT=Float32, read =false, save = false)
    xtrain = randn(FT, (tilesize, tilesize, 1, 4000))
    xtrain = reshape(xtrain, tilesize, tilesize, 1, :)
    xtest = randn(FT, (tilesize, tilesize, 1, 1000))
    xtest = reshape(xtest, tilesize, tilesize, 1, :)

    if save
        maxtrain = maximum(xtrain, dims=(1, 2, 4))
        mintrain = minimum(xtrain, dims=(1, 2, 4))
        Δ = maxtrain .- mintrain
        # To prevent dividing by zero
        Δ[Δ .== 0] .= FT(1)
        scaling = StandardScaling{FT}(mintrain, Δ)
        JLD2.save_object(preprocess_params_file, scaling)
    elseif read
        scaling = JLD2.load_object(preprocess_params_file)
    end
    xtrain .= apply_preprocessing(xtrain, scaling)
    # apply the same rescaler as on training set
    xtest .= apply_preprocessing(xtest, scaling)

    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end