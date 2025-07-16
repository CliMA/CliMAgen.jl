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
using CliMAgen

# run from giorgni2d
include("../utils_data.jl") # for data loading
include("../utils_analysis.jl") # for data loading
include("dataloader.jl") # for data loading

function run_analysis(params, f_path, savedir; FT=Float32)
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    
    batchsize = params.data.batchsize
    fraction = params.data.fraction

    inchannels = params.model.noised_channels
    nsamples = params.sampling.nsamples
    nimages = params.sampling.nimages
    batchsize = nsamples
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    rngseed > 0 && Random.seed!(rngseed)

    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        @info "Sampling on CPU"
    end

    train_dataloader, test_dataloader = get_data(
        f_path, "snapshots", batchsize)

    xtrain = first(train_dataloader)
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model_smooth)
    
    # sample from the trained model
    resolution = size(xtrain)[1]
    time_steps, Δt, init_x = setup_sampler(
        model,
        device,
        resolution,
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

    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png")
    qq_plot(xtrain, samples, savedir, "qq_plot.png")
    spectrum_plot(xtrain, samples, savedir, "mean_spectra.png")

    # create plots with nimages images of sampled data and training data
    for ch in 1:inchannels
        heatmap_grid(samples[:, :, [ch], 1:nimages], ch, savedir, "$(sampler)_images_$(ch).png")
        heatmap_grid(xtrain[:, :, [ch], 1:nimages], ch, savedir, "train_images_$(ch).png")
    end
    ncum = 10
    cum_x = zeros(ncum)
    cum_samples = zeros(ncum)
    for i in 1:ncum
        cum_x[i] = cumulant(xtrain[:],i)
        cum_samples[i] = cumulant(samples[:],i)
    end
    scatter(cum_x,label="Data",xlabel="Cumulants")
    scatter!(cum_samples, label="Gen")
    savefig(joinpath(savedir,"cumulants.png"))

    stephist(xtrain[:],normalize=:pdf,label="Data")
    stephist!(samples[:],normalize=:pdf, label="Gen")
    savefig(joinpath(savedir,"pdfs.png"))

    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)

    hfile = h5open(f_path[1:end-5] * "_analysis.hdf5", "w")
    hfile["data cumulants"] = cum_x
    hfile["generative cumulants"] = cum_samples
    hfile["samples"] = samples
    close(hfile)
end

function main(;model_toml="Model.toml", experiment_toml="Experiment.toml")
    FT = Float32
    toml_dict = TOML.parsefile(model_toml)
    α = FT(toml_dict["param_group"]["alpha"])
    β = FT(toml_dict["param_group"]["beta"])
    γ = FT(toml_dict["param_group"]["gamma"])
    σ = FT(toml_dict["param_group"]["sigma"])
    f_path = "/nobackup1/sandre/ResponseFunctionTrainingData/data_$(α)_$(β)_$(γ)_$(σ).hdf5"

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    savedir = "/nobackup1/sandre/ResponseFunctionRestartFiles/$(params.experiment.savedir)_$(α)_$(β)_$(γ)_$(σ)"
    # set up directory for saving checkpoints
    !ispath(savedir) && mkpath(savedir)
    run_analysis(params, f_path, savedir; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(; model_toml = ARGS[1], experiment_toml=ARGS[2])
end
