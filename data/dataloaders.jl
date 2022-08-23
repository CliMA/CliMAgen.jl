get_dataloader(ds::Turbulence2D; kwargs...) = get_dataloader_turbulence_2d(obtain_local_dataset_path(ds); kwargs...)
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
