

function load_data_from_file(files)
    # load data
    physical_sigmas = []
    physical_mus = []
    oldfields = []
    sigma_maxs = []
    tas_rescaleds = []
    for file in files
        FT = Float32
        hfile = h5open("/nobackup1/users/sandre/GaussianEarth/" * file, "r")
        physical_sigma = FT.(read(hfile["std"]) * extra_scale) 
        physical_mu = FT.(read(hfile["mean"]))
        oldfield = FT.(read(hfile["timeseries"]) / extra_scale) 
        sigma_max =  FT(read(hfile["max distance"] )  / extra_scale)
        tas_rescaled = read(hfile["tasrescaled"])
        close(hfile)
        push!(physical_sigmas, physical_sigma)
        push!(physical_mus, physical_mu)
        push!(oldfields, oldfield)
        push!(sigma_maxs, sigma_max)
        push!(tas_rescaleds, tas_rescaled)
    end
    M, N, _, L = size(oldfields[1])
    oldfield = zeros(M, N, length(files), L)
    for i in eachindex(files)
        oldfield[:, :, i:i, :] .= oldfields[i]
    end
    physical_mu = zeros(1, 1, length(files), 1)
    physical_sigma = zeros(1, 1, length(files), 1)
    for i in eachindex(files)
        physical_mu[1, 1, i, 1] = physical_mus[i]
        physical_sigma[1, 1, i, 1] = physical_sigmas[i]
    end
    sigma_max = sum(sigma_maxs)
    return (; physical_sigma, physical_mu, oldfield, sigma_max)
end