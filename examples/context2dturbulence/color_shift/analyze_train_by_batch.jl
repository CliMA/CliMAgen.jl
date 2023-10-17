using BSON
using CUDA
## Script for computing metrics of interest on the training data ##

using Flux
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using DelimitedFiles
using StatsBase
using HDF5

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function obtain_train_dl(params, wavenumber, FT)
    # unpack params
    savedir = params.experiment.savedir
    batchsize = 25
    resolution = params.data.resolution
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels
    dl, _ =  get_data_context2dturbulence(
        batchsize;
        resolution = resolution,
        wavenumber = wavenumber,
        fraction = fraction,
        standard_scaling = standard_scaling,
        FT=FT,
        read=true,
        preprocess_params_file=preprocess_params_file
    )
    
    return dl
end

function main(nbatches)
    resolution = 512
    experiment_toml = "experiments/Experiment_single_wavenumber_$(resolution).toml"
    @info(experiment_toml)
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    batchsize = params.data.batchsize
    wavenumber::FT = params.data.wavenumber

    rngseed = params.experiment.rngseed
    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    dl  = obtain_train_dl(params, wavenumber, FT)
 
    noised_channels = params.model.noised_channels
    savedir = params.experiment.savedir
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    scaling = JLD2.load_object(preprocess_params_file)
    stats_savedir = string("stats/train_$(resolution)")
    !ispath(stats_savedir) && mkpath(stats_savedir)
    filenames = [joinpath(stats_savedir, "statistics_ch1_$wavenumber.csv"),joinpath(stats_savedir, "statistics_ch2_$wavenumber.csv")]
    batch_id = 1
    for batch in dl
        @info batch_id
        @info size(batch)
        # revert to real space using the inverse preprocessing step
        batch .= invert_preprocessing(batch, scaling)

        # compute metrics of interest
        train_means =  mapslices(Statistics.mean, batch, dims=[1, 2])
        train_κ2 = Statistics.var(batch, dims = (1,2))
        train_κ3 = mapslices(x -> StatsBase.cumulant(x[:],3), batch, dims=[1, 2])
        train_κ4 = mapslices(x -> StatsBase.cumulant(x[:],4), batch, dims=[1, 2])
        train_spectra = mapslices(x -> hcat(power_spectrum2d(x)[1]), batch, dims =[1,2])
        if batch_id == 1
            #Save
            fname = joinpath(stats_savedir, "samples.hdf5")
            fid = h5open(fname, "w")
            fid[string("samples")] = cpu(batch)
            close(fid)
        end

        for ch in 1:noised_channels
            output = hcat(train_means[1,1,ch,:],train_κ2[1,1,ch,:], train_κ3[1,1,ch,:],train_κ4[1,1,ch,:], transpose(train_spectra[:,1,ch,:]))
            open(filenames[ch], "a") do io
                writedlm(io, output, ',')
            end
        end
        batch_id = batch_id+1
    end
    
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse(Int64, ARGS[1]))
end
