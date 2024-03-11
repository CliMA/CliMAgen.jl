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
using CliMAgen: expand_dims, MeanSpatialScaling, StandardScaling, apply_preprocessing

include("TimeseriesData.jl")

#=
function get_data(f_path, f_variable, batchsize;
    f = 1.0,
    FT=Float32,
    rng=Random.GLOBAL_RNG,)

    fid = h5open(f_path, "r")
    features = read(fid, f_variable)
    close(fid)

    f = 0.5
    N = size(features)[end]
    N_train = round(Int, N*f)
    # data needs to be continiguous in time with respec to the last dimension
    xtrain = features[:, :, :, 1:N_train];
    xtest = features[:, :, :, N_train+1:end];
    xtrain = MLUtils.shuffleobs(rng, xtrain)
    loader_train = DataLoaders.DataLoader(TimeseriesData(xtrain), batchsize)
    loader_test = DataLoaders.DataLoader(TimeseriesData(xtest), batchsize)
    return (; loader_train, loader_test)
end
=#

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
    features = features[:,:,:, shuffle(1:n_data)]
    
    # splitting
    if split == :train
        features, _ = MLUtils.splitobs(features, at=0.8) # normally at 0.8
    elseif split == :test
        _, features = MLUtils.splitobs(features, at=0.8) # normally at 0.8
    end
    return Dataset(split, Tx.(features))
end

function get_data(f_path, f_variable, batchsize;
    f = 1.0,
    FT=Float32,
    rng=Random.GLOBAL_RNG,)

    xtrain = Dataset(:train, f_path, f_variable; f = f, Tx=FT)[:];
    xtest = Dataset(:test, f_path, f_variable; f = f, Tx=FT)[:];
    xtrain = MLUtils.shuffleobs(rng, xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)
    return (; loader_train, loader_test)
end