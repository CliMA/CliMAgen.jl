module GetData

using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using HDF5
using ProgressBars
using CliMAgen
using JLD2
using FFTW
using MLDatasets, MLUtils, Images, DataLoaders, Statistics
using CliMADatasets
using CliMAgen: expand_dims, MeanSpatialScaling, StandardScaling, apply_preprocessing

export get_data


struct Dataset <: MLDatasets.UnsupervisedDataset
    split::Symbol
    features::Array{}
end


function Dataset(split::Symbol, f_path, f_variable; f = 1.0, Tx = Float32)
    # check
    @assert split ∈ [:train, :test]
    features_path = f_path
    # loading
    fid = h5open(features_path, "r")
    features = read(fid, f_variable)
    close(fid)
    
    n_observations = size(features)[end]
    n_data = Int(round(n_observations*f))
    features = features[:,:,1:1:n_data]
    
    # splitting
    if split == :train
        features, _ = MLUtils.splitobs(features, at=0.8)
    elseif split == :test
        _, features = MLUtils.splitobs(features, at=0.8)
    end
    return Dataset(split, Tx.(features))
end

function get_data(f_path, f_variable, batchsize;
    f = 1.0,
    FT=Float32,
    rng=Random.GLOBAL_RNG)

    rawtrain = Dataset(:train, f_path, f_variable; f = f, Tx=FT)[:];
    rawtest = Dataset(:test, f_path, f_variable; f = f, Tx=FT)[:];

    # Create train and test datasets
    nobs_train = size(rawtrain)[end]
    img_size = size(rawtrain)[1:end-1]
    xtrain = reshape(rawtrain, (img_size..., 1, nobs_train))
    nobs_test = size(rawtest)[end]
    xtest = reshape(rawtest, (img_size..., 1, nobs_test))
    xtrain = MLUtils.shuffleobs(rng, xtrain)
    println(batchsize)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)
    return (; loader_train, loader_test)
end
end