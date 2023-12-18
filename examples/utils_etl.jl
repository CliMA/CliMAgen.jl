using Flux
using StatsBase
using Statistics
using CliMAgen
using CUDA
using Images
using Random
using DelimitedFiles
using DataFrames
using HDF5


"""
    store_cumulants(data, gen, savepath, plotname; FT=Float32, logger=nothing)
"""
function store_cumulants(data, gen; savepath=nothing, FT=Float32, logger=nothing)
    statistics = (Statistics.mean, Statistics.var, x -> StatsBase.cumulant(x[:], 3), x -> StatsBase.cumulant(x[:], 4))

     # CDF of the generated data for each channel and each statistics
    gen = gen |> Flux.cpu
    gen_results = mapslices.(statistics, Ref(gen), dims=[1, 2])
    gen_results = cat(gen_results..., dims=ndims(gen) - 2)
    sort!(gen_results, dims=ndims(gen_results))

    # CDF of the  data for each channel and each statistics
    data_results = mapslices.(statistics, Ref(data), dims=[1, 2])
    data_results = cat(data_results..., dims=ndims(data) - 2)
    sort!(data_results, dims=ndims(data_results)) 
        
    # store data in dataframe
    ncumulants = 4
    types = [:data, :gen]
    inchannels = size(data)[end-1]
    df_out = []
    for cumu in 1:ncumulants
        for type in types
            for ch in 1:inchannels
                if type == :data
                    df = DataFrame(values = data_results[1, cumu, ch, :])
                elseif type == :gen
                    df = DataFrame(values = gen_results[1, cumu, ch, :])
                end
                df.isreal .= type == :gen ? false : true
                df.channel .= ch
                df.cumulant .= cumu
                push!(df_out, df)
            end
        end
    end
    df_out = vcat(df_out...)

    if savepath !== nothing
      filepath = joinpath(savepath, "cumulants.csv")
      writedlm(filepath, Iterators.flatten(([names(df_out)], eachrow(df_out))), ',')
    end

    return df_out
end

"""
    store_samples(data, gen, savepath, plotname; FT=Float32, logger=nothing)
"""
function store_samples(data, gen, savepath; FT=Float32, logger=nothing)
    # clip samples to [0, 1] range
    @. data = max(0, data)
    @. data = min(1, data)
    @. gen = max(0, gen)
    @. gen = min(1, gen)

    # store data in dataframe
    types = [:data, :gen]
    inchannels = size(data)[end-1]
    nsamples = size(data)[end]

    for type in types
        for ch in 1:inchannels
            for s in 1:nsamples
                if type == :data
                    filepath = joinpath(savepath, "imgs/data_ch$(ch)_$(s).png")
                    Images.save(filepath, data[:, :, ch, s])
                elseif type == :gen
                    filepath = joinpath(savepath, "imgs/gen_ch$(ch)_$(s).png")
                    Images.save(filepath, gen[:, :, ch, s])
                end
            end
        end
    end
end

"""
    read_from_hdf5(key = "generated_samples", hdf5_path="samples.hdf5")

Reads in and allocates memory to hold the data stored in `key`
in the hdf5 file found at `hdf5_path`.
"""
function read_from_hdf5(key = "generated_samples", hdf5_path="samples.hdf5")

    fid = HDF5.h5open(hdf5_path, "r")

    samples = read(fid[key])
    close(fid)

    return samples
end

"""
    drop_to_hdf5(samples; key = "generated_samples", hdf5_path="samples.hdf5")

Saves `samples` to the hdf5 file found at `hdf5_path` under key `key`.
"""
function drop_to_hdf5(samples; key = "generated_samples", hdf5_path="samples.hdf5")
    # set up HDF5
    hdf5_path = joinpath(savedir, samples_file)
    fid = HDF5.h5open(hdf5_path, "w")
    fid["generated_samples"] = samples
    close(fid)
end
