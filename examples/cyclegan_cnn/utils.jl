using Flux: DataLoader
using HDF5
using MLUtils

FT = Float32

function get_dataloader(path; field="moisture", split_ratio=0.5, batch_size=1, nsamples=100, dev=cpu)
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

    # fix data types and bring to device
    X_lo_res = FT.(X_lo_res[:, :, :, 1:nsamples]) |> dev
    X_hi_res = FT.(X_hi_res[:, :, :, 1:nsamples]) |> dev

    data_training, data_validation = MLUtils.splitobs((X_lo_res, X_hi_res), at=split_ratio)
    loader_training = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batch_size, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end
