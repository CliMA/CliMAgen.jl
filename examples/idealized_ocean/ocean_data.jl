using JLD2
using FFTW
using MLDatasets, MLUtils, Images, DataLoaders, Statistics
using CliMADatasets
using CliMAgen: expand_dims
using DataDeps
using Random

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl")) 

# Initial conditions from 10 years evolved 1/12th degree simulation
path = "https://dl.dropboxusercontent.com/scl/fi/767aqumfb40aumyqh8m54/ocean_surface_data_8.jld2?rlkey=cgdbdrg54o8lpyloibpdkwjy4&dl=0"

dh = DataDep("ocean_data", "1/8th degree resolution surface fields, channels are (1) u, (2) v, (3) w, (5) buoyancy", path)
DataDeps.register(dh)

datadep"ocean_data"

function load_ocean_data(; irange = 1:526, jrange = 1:1134) 
    filepath = datadep"ocean_data/ocean_surface_data_8.jld2"
    return jldopen(filepath)["data"][irange, jrange, :, :]
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
                        sigma_max_comp = true,
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

    if sigma_max_comp
        @info "computing sigma max"
        σmax = compute_sigma_max(xtrain)
        println("sigma max is ", σmax)
    end

    return (; loader_train, loader_test) 
end