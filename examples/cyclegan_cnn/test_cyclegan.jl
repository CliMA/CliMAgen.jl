using BSON
using CUDA
using Dates
using Flux
using Flux: params, update!
using FluxTraining
using HDF5
using MLUtils
using Statistics: mean

using Downscaling: PatchDiscriminator, UNetGenerator


function get_dataloader(path; field="vorticity", split_ratio=0.5, batchsize=1)
    fid = h5open(path, "r")
    X_lo_res = read(fid, "low_resolution/" * field)
    X_hi_res = read(fid, "high_resolution/" * field)
    close(fid)

    # TODO: needs to be handled by a data transfomer object, e.g.
    # by using a MinMaxScaler object
    # normalize data
    X_lo_res .-= (maximum(X_lo_res) + minimum(X_lo_res)) / 2
    X_lo_res ./= (maximum(X_lo_res) - minimum(X_lo_res)) / 2
    X_hi_res .-= (maximum(X_hi_res) + minimum(X_hi_res)) / 2
    X_hi_res ./= (maximum(X_hi_res) - minimum(X_hi_res)) / 2

    data_training, data_validation = MLUtils.splitobs((X_lo_res, X_hi_res), at=split_ratio)
    loader_training = Flux.DataLoader(data_training, batchsize=batchsize, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batchsize, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

function test()
end

test()
