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
    toml_dict = TOML.parsefile("correlated_ou/trj_score.toml")
    alpha = toml_dict["param_group"]["alpha"]
    beta = toml_dict["param_group"]["beta"]
    gamma = toml_dict["param_group"]["gamma"]
    sigma = toml_dict["param_group"]["sigma"]

    # read experiment parameters from file
    savedir = "output_$(alpha)_$(beta)_$(gamma)_$(sigma)"
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    batchsize = params.data.batchsize
    fraction = params.data.fraction
    inchannels = params.model.noised_channels
    nsamples = batchsize
    nsteps = params.sampling.nsteps
    tilesize_sampling = params.sampling.tilesize
    sampler = params.sampling.sampler

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
    f_path = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
    f_variable = "timeseries"
    # set up dataset
    dataloaders,_ = get_data(
        f_path, f_variable,batchsize;
        f = fraction,
        FT=Float32,
        rng=Random.GLOBAL_RNG
    )
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
    
    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png", logger=logger)
    qq_plot(xtrain, samples, savedir, "qq_plot.png", logger=logger)
    spectrum_plot(xtrain, samples, savedir, "mean_spectra.png", logger=logger)

    # create plots with nimages images of sampled data and training data
    # for ch in 1:inchannels
    #     heatmap_grid(samples[:, :, [ch], 1:nimages], ch, savedir, "$(sampler)_images_$(ch).png")
    #     heatmap_grid(xtrain[:, :, [ch], 1:nimages], ch, savedir, "train_images_$(ch).png")
    # end
    
    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    
end

function run_analysis2(params; FT=Float32, logger=nothing)
    toml_dict = TOML.parsefile("correlated_ou/trj_score.toml")
    alpha = toml_dict["param_group"]["alpha"]
    beta = toml_dict["param_group"]["beta"]
    gamma = toml_dict["param_group"]["gamma"]
    sigma = toml_dict["param_group"]["sigma"]

    # read experiment parameters from file
    savedir = "output_$(alpha)_$(beta)_$(gamma)_$(sigma)"
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    batchsize = params.data.batchsize
    fraction = params.data.fraction
    inchannels = params.model.noised_channels
    nsamples = params.sampling.nsamples
    nsteps = params.sampling.nsteps
    tilesize_sampling = params.sampling.tilesize
    sampler = params.sampling.sampler

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
    f_path = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
    f_variable = "timeseries"
    # set up dataset
    dataloaders,_ = get_data(
        f_path, f_variable,batchsize;
        f = fraction,
        FT=Float32,
        rng=Random.GLOBAL_RNG
    )
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

    file_path = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
    hfile = h5open(file_path, "r") 
    x = read(hfile["timeseries"])
    close(hfile)
    
    cum_x = zeros(10)
    cum_samples = zeros(10)
    for i in 1:10
        cum_x[i] = cumulant(reshape(x,(64*size(x)[3])),i)
        cum_samples[i] = cumulant(reshape(samples,(64*size(samples)[4])),i)
    end
    scatter(cum_x,label="Data",xlabel="Cumulants")
    scatter!(cum_samples, label="Gen")
    savefig("output_$(alpha)_$(beta)_$(gamma)_$(sigma)/cumulants.png")

    stephist(reshape(x,(64*size(x)[3])),normalize=:pdf,label="Data",xlims=(-0.95,0.95),ylims=(0,5))
    stephist!(reshape(samples,(64*size(samples)[4])),normalize=:pdf, label="Gen",xlims=(-0.95,0.95),ylims=(0,5))
    savefig("output_$(alpha)_$(beta)_$(gamma)_$(sigma)/pdfs.png")
end

experiment_toml="correlated_ou/Experiment.toml"
FT = Float32
# read experiment parameters from file
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
logger = nothing
run_analysis1(params; FT=FT, logger=logger);
run_analysis2(params; FT=FT, logger=logger);
