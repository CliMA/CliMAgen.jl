using JLD2
using FFTW
using MLDatasets, MLUtils, Images, DataLoaders, Statistics
using CliMADatasets
using CliMAgen: expand_dims
using Random

"""
Helper function that creates uniform images and returns loaders.
"""
function get_data_uniform(batchsize, std, ndata; size=32, FT=Float32)
    train_means = FT.(randn(ndata)*std)
    test_means = FT.(randn(ndata)*std)
    xtrain = zeros(FT, (size, size, 1, ndata)) .+ expand_dims(train_means, 3)

    maxtrain = maximum(xtrain, dims=(1, 2, 4))
    mintrain = minimum(xtrain, dims=(1, 2, 4))
    xtrain = @. 2(xtrain - mintrain) / (maxtrain - mintrain) - 1
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    
    xtest = zeros(FT, (size, size, 1, ndata)) .+ expand_dims(test_means, 3)
    xtest = @. 2(xtest - mintrain) / (maxtrain - mintrain) - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)
    return (; loader_train, loader_test)
end

"""
Helper function that loads MNIST images and returns loaders.
"""
function get_data_mnist(batchsize; tilesize=32, FT=Float32)
    xtrain, _ = MLDatasets.MNIST(:train; Tx=FT)[:]
    xtrain = Images.imresize(xtrain, (tilesize, tilesize))
    xtrain = reshape(xtrain, tilesize, tilesize, 1, :)
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.MNIST(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (tilesize, tilesize))
    xtest = reshape(xtest, tilesize, tilesize, 1, :)
    xtest = @. 2xtest - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads FashionMNIST images and returns loaders.
"""
function get_data_fashion_mnist(batchsize; tilesize=32, FT=Float32)
    xtrain, _ = MLDatasets.FashionMNIST(:train; Tx=FT)[:]
    xtrain = reshape(xtrain, size(xtrain)[1], size(xtrain)[2], 1, :)
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.FashionMNIST(:test; Tx=FT)[:]
    xtest = reshape(xtest, size(xtest)[1], size(xtest)[1], 1, :)
    xtest = @. 2xtest - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    # wrap iterators for resizing
    ratio_x = tilesize/size(xtrain)[1]
    ratio_y = tilesize/size(xtrain)[2]
    loader_train = Iterators.map(x -> mapslices(y -> Images.imresize(y, ratio=(ratio_x, ratio_y)), x, dims=(1,2)), loader_train)
    loader_test = Iterators.map(x -> mapslices(y -> Images.imresize(y, ratio=(ratio_x, ratio_y)), x, dims=(1,2)), loader_test)

    return (; loader_train, loader_test)
end

"""
Helper function that loads CIFAR10 images and returns loaders.
"""
function get_data_cifar10(batchsize; tilesize=32, FT=Float32)
    xtrain, _ = MLDatasets.CIFAR10(:train; Tx=FT)[:]
    xtrain = reshape(xtrain, tilesize, tilesize, 3, :)
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest, _ = MLDatasets.CIFAR10(:test; Tx=FT)[:]
    xtest = Images.imresize(xtest, (tilesize, tilesize))
    xtest = reshape(xtest, tilesize, tilesize, 3, :)
    xtest = @. 2xtest - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads 2d turbulence images and returns loaders.
"""
function get_data_celeba_hq(batchsize; resolution=32, gender=:male, FT=Float32)
    xtrain = CliMADatasets.CelebAHQ(:train; resolution=resolution, gender=gender, Tx=FT)[:]

    # bring data to [-1, 1] range
    xtrain = @. 2xtrain - 1
    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)

    xtest = CliMADatasets.CelebAHQ(:test; resolution=resolution, gender=gender, Tx=FT)[:]

    # bring data to [-1, 1] range
    xtrain = @. 2xtrain - 1
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that loads 2d turbulence images and returns loaders.
"""
function get_data_2dturbulence(batchsize;
                               width=(32, 32),
                               stride=(32, 32),
                               standard_scaling=true,
                               kernel_std=0,
                               bias_amplitude=0,
                               bias_wn=1,
                               FT=Float32)
    xtrain = CliMADatasets.Turbulence2D(:train; resolution=:high, Tx=FT)[:]
    xtrain = tile_array(xtrain, width[1], width[2], stride[1], stride[2])

    xtest = CliMADatasets.Turbulence2D(:test; resolution=:high, Tx=FT)[:]
    xtest = tile_array(xtest, width[1], width[2], stride[1], stride[2])

    if kernel_std > 0
        kernel = Kernel.gaussian(kernel_std)
        filter(img) = imfilter(img, kernel)
        xtrain = mapslices(filter, xtrain, dims = (1,2))
        xtest = mapslices(filter, xtest, dims = (1,2))
    end

    if bias_amplitude > 0
        nx, ny = size(xtrain)[1:2]
        amp = maximum(xtrain, dims=(1,2,4)) - minimum(xtrain, dims=(1,2,4))

        xx = FT.(LinRange(0, 1, nx)) * ones(FT, ny)'
        yy = ones(FT, nx) * FT.(LinRange(0, 1, ny))'

        bias_field = sin.(2π .* xx .* bias_wn) .* sin.(2π .* yy .* bias_wn)
        xtrain = xtrain .* (1 .+ amp .* bias_amplitude .* bias_field)
        xtest = xtest .* (1 .+ amp .* bias_amplitude .* bias_field)
    end
    
    if standard_scaling
        # perform a standard minmax scaling
        maxtrain = maximum(xtrain, dims=(1, 2, 4))
        mintrain = minimum(xtrain, dims=(1, 2, 4))
        xtrain = @. 2(xtrain - mintrain) / (maxtrain - mintrain) - 1
        # apply the same rescaler as on training set
        xtest = @. 2(xtest - mintrain) / (maxtrain - mintrain) - 1
    else
        #scale means and spatial variations separately
        x̄ = mean(xtrain, dims=(1, 2))
        maxtrain_mean = maximum(x̄, dims=4)
        mintrain_mean = minimum(x̄, dims=4)
        Δ̄ = maxtrain_mean .- mintrain_mean
        x̄̃ = @. 2(x̄ -  mintrain_mean) / Δ̄ - 1
        
        xp = xtrain .- x̄
        maxtrain_p = maximum(xp, dims=(1, 2, 4))
        mintrain_p = minimum(xp, dims=(1, 2, 4))
        Δp = maxtrain_p .- mintrain_p
        x̃p = @. 2(xp -  mintrain_p) / Δp - 1
    
        xtrain = x̄̃ .+ x̃p

         # apply the same rescaler as on training set
        x̄ = mean(xtest, dims=(1, 2))
        xp = xtest .- x̄
        x̄̃ = @. 2(x̄ - mintrain_mean) / Δ̄ - 1
        x̃p = @. 2(xp - mintrain_p) / Δp - 1

        xtest = x̄̃ .+ x̃p
    end

    xtrain = MLUtils.shuffleobs(xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
    get_data_context2dturbulence(batchsize;
                                 rng=Random.GLOBAL_RNG,
                                 resolution = 512,
                                 wavenumber = 0.0,
                                 fraction = 1.0,
                                 standard_scaling = false,
                                 FT = Float32,
                                 read = false,
                                 save = false,
                                 preprocess_params_file)

Obtains the raw data from the 2D turbulence with context dataset,
carries out a scaling of the data, and loads the data into train and test
dataloders, which are returned.

The user can pick:
- resolution:       (64 or 512)
- wavenumber:       (0 = all wavenumbers, supported for both resolutions
                    or, 2,4,8,16, supported only for 512 resolution.)
- fraction:         the amount of the data to use. Must be of the form 1/integer.
- standard_scaling: boolean indicating if standard minmax scaling is used
                    or if minmax scaling of the mean and spatial variations
                    are both implemented.
- FT:               the float type of the model
- read:             a boolean indicating if the preprocessing parameters should be read
- save:             a boolean indicating if the preprocessing parameters should be
                    computed and read.
- preprocess_params:filename where preprocessing parameters are stored or read from.

If a resolution of 64 is chosen, the raw data is upsampled to 512x512 using
nearest-neighbors, and then low-pass filtered.
"""
function get_data_context2dturbulence(batchsize;
                                      rng=Random.GLOBAL_RNG,
                                      resolution=512,
                                      wavenumber=0.0,
                                      fraction = 1.0,
                                      standard_scaling = false,
                                      FT=Float32,
                                      read = false,
                                      save = false,
                                      preprocess_params_file)
    @assert xor(read, save)
    @assert resolution ∈ [512, 64]
    if resolution == 512
        @assert wavenumber ∈ FT.([0, 1, 2, 4, 8, 16])
    elseif resolution == 64
        @assert wavenumber ∈ FT.([0, 1])
    end

    if wavenumber == FT(0) # Returns all the data, for every wavenumber
        xtrain = CliMADatasets.Turbulence2DContext(:train; fraction = fraction, resolution=resolution, wavenumber = :all, Tx=FT,)[:]
        xtest = CliMADatasets.Turbulence2DContext(:test; fraction = fraction, resolution=resolution, wavenumber = :all, Tx=FT,)[:]
    else # Returns data for a specific wavenumber only
        xtrain = CliMADatasets.Turbulence2DContext(:train; fraction = fraction, resolution=resolution, wavenumber = wavenumber, Tx=FT,)[:]
        xtest = CliMADatasets.Turbulence2DContext(:test; fraction = fraction, resolution=resolution, wavenumber = wavenumber, Tx=FT,)[:]
    end

    if resolution == 64
        # Upsampling
        upsample = Flux.Upsample(8, :nearest)
        xtrain_upsampled = Complex{FT}.(upsample(xtrain))
        xtest_upsampled = Complex{FT}.(upsample(xtest));
        # Upsampling produces artifacts at high frequencies, so now
        # we filter.
        fft!(xtrain_upsampled, (1,2));
        xtrain_upsampled[:,33:479,:,:] .= Complex{FT}(0);
        xtrain_upsampled[33:479,:,:,:] .= Complex{FT}(0);
        ifft!(xtrain_upsampled, (1,2))
        xtrain = real(xtrain_upsampled)

        fft!(xtest_upsampled, (1,2));
        xtest_upsampled[:,33:479,:,:] .= Complex{FT}(0);
        xtest_upsampled[33:479,:,:,:] .= Complex{FT}(0);
        ifft!(xtest_upsampled, (1,2))
        xtest = real(xtest_upsampled)
    end
    
    if save
        if standard_scaling
            maxtrain = maximum(xtrain, dims=(1, 2, 4))
            mintrain = minimum(xtrain, dims=(1, 2, 4))
            Δ = maxtrain .- mintrain
            # To prevent dividing by zero
            Δ[Δ .== 0] .= FT(1)
            scaling = StandardScaling{FT}(mintrain, Δ)
        else
            #scale means and spatial variations separately
            x̄ = mean(xtrain, dims=(1, 2))
            maxtrain_mean = maximum(x̄, dims=4)
            mintrain_mean = minimum(x̄, dims=4)
            Δ̄ = maxtrain_mean .- mintrain_mean
            xp = xtrain .- x̄
            maxtrain_p = maximum(xp, dims=(1, 2, 4))
            mintrain_p = minimum(xp, dims=(1, 2, 4))
            Δp = maxtrain_p .- mintrain_p

            # To prevent dividing by zero
            Δ̄[Δ̄ .== 0] .= FT(1)
            Δp[Δp .== 0] .= FT(1)
            scaling = MeanSpatialScaling{FT}(mintrain_mean, Δ̄, mintrain_p, Δp)
        end
        JLD2.save_object(preprocess_params_file, scaling)
    elseif read
        scaling = JLD2.load_object(preprocess_params_file)
    end
    xtrain .= apply_preprocessing(xtrain, scaling)
    # apply the same rescaler as on training set
    xtest .= apply_preprocessing(xtest, scaling)

    xtrain = MLUtils.shuffleobs(rng, xtrain)
    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    return (; loader_train, loader_test)
end

"""
Helper function that tiles an array in the first two spatial dimensions.

Tiles wrap around periodically if input width is larger than spatial size of array.

This is currently hardcoded for two spatial dimensions.
"""
function tile_array(A::AbstractArray, xwidth::Int, ywidth::Int, xstride::Int, ystride::Int)
    @assert ndims(A) == 4

    # number of tiles in x and y direction
    xsize, ysize = Base.size(A)[1:2]
    nx = floor(Int, abs(xsize - xwidth) / xstride)
    ny = floor(Int, abs(ysize - ywidth) / ystride)

    # tile up the array in spatial directions only!
    processed_data = []
    xranges = map(k -> 1+k*xstride:xwidth+k*xstride, 0:nx)
    yranges = map(k -> 1+k*ystride:ywidth+k*ystride, 0:ny)

    # if width > size of array, we wrap around periodically
    xranges = map(x -> map(y -> mod(y, xsize) != 0 ? mod(y, xsize) : xsize, x), xranges)
    yranges = map(x -> map(y -> mod(y, ysize) != 0 ? mod(y, ysize) : ysize, x), yranges)
    for (xr, yr) in Base.Iterators.product(xranges, yranges)
        push!(processed_data, A[xr, yr, :, :])
    end
    return cat(processed_data..., dims=ndims(A))
end
