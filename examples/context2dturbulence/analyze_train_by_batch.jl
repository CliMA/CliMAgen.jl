using BSON
using CUDA
using Flux
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using DelimitedFiles
using StatsBase

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))
function obtain_train_dl(params, wavenumber, FT)
    # unpack params
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels
    
    # set up dataset - we need this in order to get the context
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

function main(npixels, wavenumber; experiment_toml="Experiment.toml")
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    batchsize = params.data.batchsize

    rngseed = params.experiment.rngseed
    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    dl  = obtain_train_dl(params, wavenumber, FT)
    resolution = string(params.data.resolution)
    noised_channels = params.model.noised_channels
    savedir = params.experiment.savedir
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    scaling = JLD2.load_object(preprocess_params_file)

    stats_savedir = string("stats/",resolution,"x", resolution,"/train")
    filenames = [joinpath(stats_savedir, "train_statistics_ch1_$wavenumber.csv"),joinpath(stats_savedir, "train_statistics_ch2_$wavenumber.csv")]
    pixel_filenames = [joinpath(stats_savedir, "train_pixels_ch1_$wavenumber.csv"),joinpath(stats_savedir, "train_pixels_ch2_$wavenumber.csv")]
    # The 64x64 resolution images have been enlarged to 512x512
    # This is hardcoded
    train_pixels = zeros(FT,(512*512, noised_channels, batchsize))

    for batch in dl
        # revert to real space using the inverse preprocessing step
        batch .= invert_preprocessing(batch, scaling)

        # compute metrics of interest
        train_means =  mapslices(Statistics.mean, batch, dims=[1, 2])
        train_κ2 = Statistics.var(batch, dims = (1,2))
        train_κ3 = mapslices(x -> StatsBase.cumulant(x[:],3), batch, dims=[1, 2])
        train_κ4 = mapslices(x -> StatsBase.cumulant(x[:],4), batch, dims=[1, 2])
        train_spectra = mapslices(x -> hcat(power_spectrum2d(x)[1]), batch, dims =[1,2])

        # average instant condensation rate
        train_icr = make_icr(batch)

        # batch is 512 x 512 x 3 x batchsize/nsamples, except possibly on the last batch. 
        current_batchsize = size(batch)[end]
        train_pixels[:,:,1:current_batchsize] .= reshape(batch[:,:, 1:noised_channels, :], (prod(size(batch)[1:2]), noised_channels, current_batchsize))
        pixel_indices = StatsBase.sample(1:1:size(train_pixels)[1], npixels)


        for ch in 1:noised_channels
            # write pixel vaues to other file
            open(pixel_filenames[ch],"a") do io
                writedlm(io, transpose(train_pixels[pixel_indices, ch, 1:current_batchsize]), ',')
            end

            if ch == 1
                output = hcat(train_means[1,1,ch,:],train_κ2[1,1,ch,:], train_κ3[1,1,ch,:],train_κ4[1,1,ch,:], transpose(train_spectra[:,1,ch,:]), train_icr[1,1,ch,:])
            else
                output = hcat(train_means[1,1,ch,:],train_κ2[1,1,ch,:], train_κ3[1,1,ch,:],train_κ4[1,1,ch,:], transpose(train_spectra[:,1,ch,:]))
            end
            open(filenames[ch], "a") do io
                writedlm(io, output, ',')
            end
        end
    end
    
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse(Int64, ARGS[1]), parse(Float32, ARGS[2]); experiment_toml=ARGS[3])
end
