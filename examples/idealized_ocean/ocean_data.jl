using JLD2
using FFTW
using MLDatasets, MLUtils, Images, DataLoaders, Statistics
using CliMADatasets
using CliMAgen: expand_dims
using Random

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl")) 

function load_ocean_data(; irange = 1:526, jrange = 1:1134) 
    file = "/orcd/nese/raffaele/001/ssilvest/4Darray_Nx_Ny_channel_uvwb_Nt_float32.jld2"
    return jldopen(file)["data"][irange, jrange, :, :]
end

function compute_sigma_max(x; )
    n_obs = size(x)[end]
    max_distance = 0
    for i in 1:n_obs
        for j in i+1:n_obs
            distance = sqrt(sum((x[:,:,:,i] .- x[:,:,:,j]).^2))
            max_distance = max(max_distance, distance)
        end
    end
    return max_distance
end

function get_data_ocean(batchsize; 
                        irange = 1:526, 
                        jrange = 1:1134, 
                        train_fraction = 0.8, 
                        channels = 1:4,
                        FT = Float32)
    
    data = load_ocean_data(; irange, jrange)
    train_range = 1:floor(Int, size(data, 4) * train_fraction)
    test_range  = train_range[end]:size(data, 4)
    xtrain  = FT.(data[:, :, channels, train_range])
    xtest   = FT.(data[:, :, channels, test_range])

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

    # apply the same rescaler as on training set
    xtrain .= apply_preprocessing(xtrain, scaling)
    # apply the same rescaler as on training set
    xtest .= apply_preprocessing(xtest, scaling)

    loader_train = DataLoaders.DataLoader(xtrain, batchsize)
    loader_test = DataLoaders.DataLoader(xtest, batchsize)

    @info "computing sigma max"
    σmax = compute_sigma_max(xtrain)
    println("sigma max is ", σmax)

    return (; loader_train, loader_test) 
end