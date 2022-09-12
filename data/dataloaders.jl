get_dataloader(ds::Turbulence2D; kwargs...) = get_dataloader_turbulence_2d(obtain_local_dataset_path(ds); kwargs...)
get_dataloader(ds::Turbulence2DComplete; kwargs...) = get_dataloader_turbulence_2d_complete(obtain_local_dataset_path(ds); kwargs...)
get_dataloader(ds::KuramotoSivashinsky; kwargs...) = get_dataloader_ks(obtain_local_dataset_path(ds); kwargs...)

function get_dataloader_ks(path; split_ratio=0.8, batch_size::Int=4, nsamples::Int=4000, augmentation_factor::Int=4, rng=Random.MersenneTwister(123))
    fid = h5open(path, "r")
    X_lo_res = read(fid, "low_resolution/u")
    X_hi_res = read(fid, "high_resolution/u")
    close(fid)

    # restrict to first nsamples
    reduced_nsamples = div(nsamples, augmentation_factor)
    X_lo_res = X_lo_res[:, :, 10*100:100:(10+reduced_nsamples-1)*100]
    X_hi_res = X_hi_res[:, :, 10*100:100:(10+reduced_nsamples-1)*100]

    # augment the data by creating pairs at random
    random_indices = StatsBase.sample(rng, Array(1:1:reduced_nsamples), reduced_nsamples * augmentation_factor * 2, replace=true)
    X_lo_res = X_lo_res[:, :, random_indices[1:reduced_nsamples*augmentation_factor]]
    X_hi_res = X_hi_res[:, :, random_indices[reduced_nsamples*augmentation_factor+1:end]]

    # Create the noise to be inserted in the middle of the resnet layers
    input_size = size(X_lo_res)
    resnet_block_input_size = (div(input_size[1], 4), 256, input_size[3])
    noise = 2 .* rand(rng, resnet_block_input_size...) .- 1

    # split data
    data_training, data_validation = MLUtils.splitobs((X_lo_res, X_hi_res, noise), at=split_ratio)

    # TODO: needs to be handled by a data transfomer object, e.g.
    # by using a MinMaxScaler
    # normalize data
    X_lo_res_train, X_hi_res_train, noise_train = data_training
    X_lo_res_val, X_hi_res_val, noise_val = data_validation
    lowest, highest = extrema(X_lo_res_train)
    X_lo_res_train = @. 2 * (X_lo_res_train - lowest) / (highest - lowest) - 1
    X_lo_res_val = @. 2 * (X_lo_res_val - lowest) / (highest - lowest) - 1
    lowest, highest = extrema(X_hi_res_train)
    X_hi_res_train = @. 2 * (X_hi_res_train - lowest) / (highest - lowest) - 1
    X_hi_res_val = @. 2 * (X_hi_res_val - lowest) / (highest - lowest) - 1
    data_training = (X_lo_res_train, X_hi_res_train, noise_train)
    data_validation = (X_lo_res_val, X_hi_res_val, noise_val)

    loader_training = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batch_size, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end

function get_dataloader_turbulence_2d(path; split_ratio=0.5, batch_size::Int=1, nsamples::Int=1000, augmentation_factor::Int=4, rng=Random.MersenneTwister(123))
    fid = h5open(path, "r")
    X_lo_res = read(fid, "low_resolution/moisture/")
    X_hi_res = read(fid, "high_resolution/moisture/")
    close(fid)

    # restrict to first nsamples
    reduced_nsamples = div(nsamples, augmentation_factor)
    X_lo_res = X_lo_res[:, :, :, 1:reduced_nsamples]
    X_hi_res = X_hi_res[:, :, :, 1:reduced_nsamples]

    # augment the data by creating pairs at random
    random_indices = StatsBase.sample(rng, Array(1:1:reduced_nsamples), reduced_nsamples * augmentation_factor * 2, replace=true)
    X_lo_res = X_lo_res[:, :, :, random_indices[1:reduced_nsamples*augmentation_factor]]
    X_hi_res = X_hi_res[:, :, :, random_indices[reduced_nsamples*augmentation_factor+1:end]]

    # Create the noise to be inserted in the middle of the resnet layers
    input_size = size(X_lo_res)
    resnet_block_input_size = (div(input_size[1], 4), div(input_size[2], 4), 256, input_size[4])
    noise = 2 .* rand(rng, resnet_block_input_size...) .- 1

    # split data
    data_training, data_validation = MLUtils.splitobs((X_lo_res, X_hi_res, noise), at=split_ratio)

    # TODO: needs to be handled by a data transfomer object, e.g.
    # by using a MinMaxScaler
    # normalize data
    X_lo_res_train, X_hi_res_train, noise_train = data_training
    X_lo_res_val, X_hi_res_val, noise_val = data_validation
    lowest, highest = extrema(X_lo_res_train)
    X_lo_res_train = @. 2 * (X_lo_res_train - lowest) / (highest - lowest) - 1
    X_lo_res_val = @. 2 * (X_lo_res_val - lowest) / (highest - lowest) - 1
    lowest, highest = extrema(X_hi_res_train)
    X_hi_res_train = @. 2 * (X_hi_res_train - lowest) / (highest - lowest) - 1
    X_hi_res_val = @. 2 * (X_hi_res_val - lowest) / (highest - lowest) - 1
    data_training = (X_lo_res_train, X_hi_res_train, noise_train)
    data_validation = (X_lo_res_val, X_hi_res_val, noise_val)

    loader_training = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batch_size, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end

function get_dataloader_turbulence_2d_complete(
    path;
    split_ratio=0.875,
    batch_size::Int=32,
    mode=:high,
    variable=:moisture,
    xwidth=32,
    ywidth=32,
    xstride=32,
    ystride=32
)
    fid = h5open(path, "r")
    if mode == :high
        data = read(fid, "high_resolution/")
    elseif mode == :low
        data = read(fid, "low_resolution/")
    end
    close(fid)

    # toggle moist or vorticity
    if variable == :moisture
        data = data[:, :, [1], :]
    elseif variable == :vorticity
        data = data[:, :, [2], :]
    elseif variable == :all
    end

    # number of tiles in x and y direction
    xsize, ysize = size(data)[1:2]
    nx = floor(Int, (xsize - xwidth) / xstride)
    ny = floor(Int, (ysize - ywidth) / ystride)

    # tile up the images
    processed_data = []
    xranges = map(k -> 1+k*xstride:xwidth+k*xstride, 0:nx)
    yranges = map(k -> 1+k*ystride:ywidth+k*ystride, 0:ny)
    for (xr, yr) in Base.Iterators.product(xranges, yranges)
        push!(processed_data, data[xr, yr, :, :])
    end
    nb = length(size(data))
    processed_data = cat(processed_data..., dims=nb)

    # split into train and test
    data_training, data_validation = MLUtils.splitobs(processed_data, at=split_ratio)

    # TODO: needs to be handled by a data transfomer object, e.g.
    # by using a MinMaxScaler
    # normalize data
    lowest, highest = extrema(data_training)
    data_training = @. 2 * (data_training - lowest) / (highest - lowest) - 1
    data_validation = @. 2 * (data_validation - lowest) / (highest - lowest) - 1

    loader_training = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batch_size, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end
