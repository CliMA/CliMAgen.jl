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

include("TimeseriesData.jl")

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
