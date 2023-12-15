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

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))
function convert_to_symbol(string)
    if string == "strong"
        return :strong
    elseif string == "medium"
        return :medium
    elseif string == "weak"
        return :weak
    else
        @error("Nonlinearity must be weak, medium, or strong.")
    end
end
function run_analysis(params; FT=Float32)
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    
    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction = params.data.fraction
    nonlinearity = convert_to_symbol(params.data.nonlinearity) 
    standard_scaling = params.data.standard_scaling
    power_transform = params.data.transform
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

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

    # set up dataset
    dataloaders,_ = get_data_response2dturbulence(batchsize;
                                                 nonlinearity = nonlinearity,
                                                 resolution = resolution,
                                                 f = fraction,
                                                 FT=FT,
                                                 rng=Random.GLOBAL_RNG,
                                                 standard_scaling = standard_scaling,
						 power_transform = power_transform,
                                                 read = false,
                                                 save = true,
                                                 preprocess_params_file = preprocess_params_file)

    xtrain = first(dataloaders)
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    
    # sample from the trained model
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

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png")

    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain, samples, savedir, "qq_plot.png")

    # create plots for comparison of real vs. generated spectra
    spectrum_plot(xtrain, samples, savedir, "mean_spectra.png")

    # create plots with nimages images of sampled data and training data
    for ch in 1:inchannels
        heatmap_grid(samples[:, :, :, 1:nimages], ch, savedir, "$(sampler)_images_$(ch).png")
        heatmap_grid(xtrain[:, :, :, 1:nimages], ch, savedir, "train_images_$(ch).png")
    end

    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)

end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    run_analysis(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
