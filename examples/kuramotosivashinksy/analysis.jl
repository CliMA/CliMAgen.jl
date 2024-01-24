using BSON
using Flux
using CUDA
using cuDNN
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using HDF5

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function run_analysis(params; FT=Float32, k_bias=0.0f0, n_avg=1)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    n_pixels = params.data.n_pixels
    n_time = params.data.n_time
    @assert n_pixels == n_time
    inchannels = params.model.inchannels

    # set up dataset
    dl,_ = get_data_ks(batchsize,preprocess_params_file;
                       f=fraction,
                       n_pixels=n_pixels,
                       n_time=n_time,
                       standard_scaling=standard_scaling,
                       read =true,
                       save=false,
                       FT=FT
                       )
    xtrain = cat([x for x in dl]..., dims=4);
    # To use Images.Gray, we need the input to be between 0 and 1.
    # Obtain max and min here using the whole data set
    maxtrain = maximum(xtrain, dims=(1, 2, 4))
    mintrain = minimum(xtrain, dims=(1, 2, 4))

    shift = params.sampling.shift
    samplesdir = joinpath(savedir, "bias_$(FT(k_bias))_n_avg_$(n_avg)_shift_$shift")
    samples_file = "samples_initial_shift_nonzero.hdf5"#params.sampling.samples_file
    hdf5_path = joinpath(samplesdir, samples_file)
    fid = HDF5.h5open(hdf5_path, "r")
    samples =  read(fid["generated_samples"])
    lr = read(fid["likelihood_ratio"])
    close(fid) 
    outputdir = samplesdir

    # Autocorrelation code 
    # Expects a timeseries of of a scalar: of size nsteps x nbatch
    # Restrict to the first nsamples so that the uncertainties are comparable
    nsamples = size(samples)[end]
    nimages = params.sampling.nimages
    id = Int(div(n_pixels, 2))
    autocorrelation_plot(xtrain[id,:,1,1:nsamples], samples[id,:,1,:], outputdir, "autocorr.png")
    observable(x) = mean(x[id,id-div(duration,2):id+div(duration,2)-1,1,:], dims = 1)[:]
    duration = 16 # autocorrelation time is 10
    event_probability_plot(observable(xtrain), observable(samples), lr, outputdir, "event_probability_$(duration).png")

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples, outputdir, "spatial_mean_distribution.png")
    
    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain[:,:,:, 1:size(samples)[end]], samples, outputdir, "qq_plot.png")

    # create plots with nimages images of sampled data and training data
    # Rescale now using mintrain and maxtrain
    xtrain = @. (xtrain - mintrain) / (maxtrain - mintrain)
    samples = @. (samples - mintrain) / (maxtrain - mintrain)

    heatmap_grid(samples[:, :, :, 1:nimages], 1, outputdir, "gen_images_ch1.png")
    heatmap_grid(xtrain[:, :, :, 1:nimages], 1, outputdir, "train_images_ch1.png")
    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    
end

function main(; experiment_toml="Experiment.toml", k_bias=0.0f0, n_avg=1)
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    run_analysis(params; FT=FT, k_bias=FT(k_bias), n_avg=Int(n_avg))

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(;experiment_toml=ARGS[1], k_bias=parse(Float64, ARGS[2]), n_avg=parse(Int64, ARGS[3]))
end