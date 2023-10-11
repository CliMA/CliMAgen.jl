include("GetData.jl")

using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using Main.GetData: get_data
using HDF5
using CliMAgen


package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function run_analysis1(params; FT=Float32, logger=nothing)
    toml_dict = TOML.parsefile("trj_score.toml")
    alpha = toml_dict["param_group"]["alpha"]
    beta = toml_dict["param_group"]["beta"]
    gamma = toml_dict["param_group"]["gamma"]
    sigma = toml_dict["param_group"]["sigma"]

    # read experiment parameters from file
    periodic = params.model.periodic
    preprocess = params.data.preprocess
    savedir_base = params.experiment.savedir_base
    savedir = string(savedir_base, "_preprocess_$(preprocess)_periodic_$(periodic)_$(alpha)_$(beta)_$(gamma)_$(sigma)")
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    fraction = params.data.fraction
    inchannels = params.model.noised_channels
    nsamples = params.sampling.nsamples
    nimages = params.sampling.nimages
    batchsize = nsamples
    nsteps = params.sampling.nsteps
    tilesize_sampling = params.sampling.tilesize
    sampler = params.sampling.sampler
    preprocess_params_file = joinpath(savedir, "preprocess.jld2")
    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        @info "Sampling on CPU"
    end
    f_path = "data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
    f_variable = "timeseries"
    # set up dataset
    dataloaders,_ = get_data(
        f_path, f_variable,batchsize;
        f = fraction,
        FT=Float32,
        rng=Random.GLOBAL_RNG,
        preprocess = preprocess,
        preprocess_save = false, # use the one that was used and saved in training
        preprocess_params_file = preprocess_params_file)

    xtrain = first(dataloaders)
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    
    # sample from the trained model
    time_steps, Δt, init_x = setup_sampler(
        model,
        device,
        tilesize_sampling,
        inchannels;
        num_images=nsamples,
        num_steps=nsteps,
    )
    if sampler == "euler"
        samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
    elseif sampler == "pc"
        samples = predictor_corrector_sampler(model, init_x, time_steps, Δt)
    end
    samples = cpu(samples) 
    scaling = JLD2.load_object(preprocess_params_file)
    samples .= CliMAgen.invert_preprocessing(samples, scaling)
    xtrain .= CliMAgen.invert_preprocessing(xtrain, scaling)


    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png", logger=logger)
    qq_plot(xtrain, samples, savedir, "qq_plot.png", logger=logger)
    spectrum_plot(xtrain, samples, savedir, "mean_spectra.png", logger=logger)

    # create plots with nimages images of sampled data and training data
    for ch in 1:inchannels
        heatmap_grid(samples[:, :, [ch], 1:nimages], ch, savedir, "$(sampler)_images_$(ch).png")
        heatmap_grid(xtrain[:, :, [ch], 1:nimages], ch, savedir, "train_images_$(ch).png")
    end
    
    cum_x = zeros(10)
    cum_samples = zeros(10)
    for i in 1:10
        cum_x[i] = cumulant(xtrain[:],i)
        cum_samples[i] = cumulant(samples[:],i)
    end
    scatter(cum_x,label="Data",xlabel="Cumulants")
    scatter!(cum_samples, label="Gen")
    savefig(joinpath(savedir,"cumulants.png"))

    stephist(xtrain[3:4,3:4,1,:][:],normalize=:pdf,label="Data",xlims=(-0.95,0.95),ylims=(0,5))
    stephist!(samples[3:4,3:4,1,:][:],normalize=:pdf, label="Gen",xlims=(-0.95,0.95),ylims=(0,5))
    savefig(joinpath(savedir,"center_pdfs.png"))


    stephist(xtrain[[1,8],[1,8],1,:][:],normalize=:pdf,label="Data",xlims=(-0.95,0.95),ylims=(0,5))
    stephist!(samples[[1,8],[1,8],1,:][:],normalize=:pdf, label="Gen",xlims=(-0.95,0.95),ylims=(0,5))
    savefig(joinpath(savedir,"edge_pdfs.png"))

    stephist(xtrain[:],normalize=:pdf,label="Data",xlims=(-0.95,0.95),ylims=(0,5))
    stephist!(samples[:],normalize=:pdf, label="Gen",xlims=(-0.95,0.95),ylims=(0,5))
    savefig(joinpath(savedir,"pdfs.png"))

    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    
end

experiment_toml="Experiment_preprocess_periodic.toml"
FT = Float32
# read experiment parameters from file
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
logger = nothing
run_analysis1(params; FT=FT, logger=logger);
