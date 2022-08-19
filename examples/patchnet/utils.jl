using Flux: DataLoader
using HDF5
using MLUtils

FT = Float32

function get_dataloader(path; field="moisture", split_ratio=0.5, batch_size=1, nsamples=1000, dev=cpu)
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

    # make series of 4 tiles
    X_lo_res = make_tiles(assemble_again(X_lo_res))
    X_hi_res = make_tiles(assemble_again(X_hi_res))

    # fix data types and bring to device
    X_lo_res = FT.(X_lo_res[:, :, :, :, 1:nsamples]) |> dev
    X_hi_res = FT.(X_hi_res[:, :, :, :, 1:nsamples]) |> dev

    data_training, data_validation = MLUtils.splitobs((X_lo_res, X_hi_res), at=split_ratio)
    loader_training = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batch_size, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end

function assemble_again(X)
    ntiles = 16
    ntiles_sqrt = convert(Int, sqrt(ntiles))
    ndata = size(X)[end]
    nactual = div(ndata, ntiles)

    imgs = []
    for img_idx in 1:nactual
        cols = []
        for row_idx in 1:ntiles_sqrt
            rows = []
            for col_idx in 1:ntiles_sqrt
                tile_idx = (img_idx-1)*ntiles + (row_idx-1)*ntiles_sqrt + col_idx
                push!(rows, X[:, :, :, [tile_idx]])
            end
            push!(cols, cat(rows..., dims=2))
        end
        img = cat(cols..., dims=1)
        push!(imgs, img)
    end

    return cat(imgs..., dims=4)
end

function make_tiles(X)
    xwidth = 128
    ywidth = 128
    xstride = div(xwidth, 2)
    ystride = div(ywidth, 2)
    xsize, ysize = size(X)[1:2] # assumes image is square
    nxtiles = div(xsize, xstride) - 1
    nytiles = div(ysize, ystride) - 1

    tiles = []
    for n = 1: size(X)[end]
        frame = X[:, :, :, [n]]
        for i in 1:nxtiles-1
            x_idx = (i-1)*xstride
            for j in 1:nytiles-1
                y_idx = (j-1)*ystride
                series = []
                xrange = (x_idx+1):(x_idx+xwidth)
                yrange = (y_idx+1):(y_idx+ywidth)
                tmp = frame[xrange, yrange, :, :]
                push!(series, reshape(tmp, (1, size(tmp)...)))
                xrange = (x_idx+1):(x_idx+xwidth)
                yrange = (y_idx+ystride+1):(y_idx+ystride+ywidth)
                tmp = frame[xrange, yrange, :, :]
                push!(series, reshape(tmp, (1, size(tmp)...)))
                xrange = (x_idx+xstride+1):(x_idx+xstride+xwidth)
                yrange = (y_idx+1):(y_idx+ywidth)
                tmp = frame[xrange, yrange, :, :]
                push!(series, reshape(tmp, (1, size(tmp)...)))
                xrange = (x_idx+xstride+1):(x_idx+xstride+xwidth)
                yrange = (y_idx+ystride+1):(y_idx+ystride+ywidth)
                tmp = frame[xrange, yrange, :, :]
                push!(series, reshape(tmp, (1, size(tmp)...)))
                series = cat(series..., dims=1)
                push!(tiles, series)
            end
        end
    end

    return cat(tiles..., dims=5)
end
