using Flux: DataLoader
using HDF5
using MLUtils
using Random
using StatsBase

FT = Float32

function get_dataloader(path; field="moisture", split_ratio=0.5, batch_size::Int=1, nsamples::Int=250, augmentation_factor::Int=4, dev=cpu, rng = Random.MersenneTwister(123))
    fid = h5open(path, "r")
    X_lo_res = read(fid, "low_resolution/" * field)
    X_hi_res = read(fid, "high_resolution/" * field)
    close(fid)

    # TODO: needs to be handled by a data transfomer object, e.g.
    # by using a MinMaxScaler
    # normalize data
    lowest, highest = extrema(X_lo_res)
    X_lo_res = @. 2 * (X_lo_res - lowest) / (highest - lowest) - 1
    lowest, highest = extrema(X_hi_res)
    X_hi_res = @. 2 * (X_hi_res - lowest) / (highest - lowest) - 1

    
    # fix data types and restrict to first nsamples
    
    X_lo_res = FT.(X_lo_res[:, :, :, 1:nsamples])
    X_hi_res = FT.(X_hi_res[:, :, :, 1:nsamples])

    # augment the data by creating pairs at random, and push to device
    random_indices = StatsBase.sample(rng, Array(1:1:nsamples), nsamples * augmentation_factor * 2, replace = true)
    X_lo_res = X_lo_res[:, :, :, random_indices[1:nsamples * augmentation_factor]] |> dev
    X_hi_res = X_hi_res[:, :, :, random_indices[nsamples * augmentation_factor + 1:end]] |> dev

    # Create the noise to be inserted in the middle of the resnet layers
    input_size = size(X_lo_res)
    resnet_block_input_size = (div(input_size[1], 4), div(input_size[2], 4), 256, input_size[4])
    noise = randn(rng, FT, resnet_block_input_size) |> dev

    data_training, data_validation = MLUtils.splitobs((X_lo_res, X_hi_res, noise), at=split_ratio)
    loader_training = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batch_size, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end
