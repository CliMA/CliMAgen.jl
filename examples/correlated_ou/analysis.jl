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
include(joinpath(package_dir, "examples/utils_data.jl"))
include(joinpath(package_dir, "examples/utils_analysis.jl"))

function run_analysis(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    resolution = params.data.resolution
    ntime = params.data.ntime
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    inchannels = params.model.inchannels

    nsamples = params.sampling.nsamples
    ngifs = params.sampling.ngifs
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler

    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    # set up device
    if !nogpu && CUDA.has_cuda()
        dev = Flux.gpu
        @info "Sampling on GPU"
    else
        dev = Flux.cpu
        @info "Sampling on CPU"
    end

    # set up dataset
    dl, _ = get_data_correlated_ou2d_timeseries(
        nsamples;
        f = fraction,
        resolution=resolution,
        ntime=ntime,
        FT=Float32,
        standard_scaling = standard_scaling,
        read = true,
        save = false,
        preprocess_params_file,
        rng=Random.GLOBAL_RNG
    )
    xtrain, dl = Iterators.peel(dl)
    nspatial=3

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = dev(model)

    # sample from the trained model
    time_steps, Δt, init_x = setup_sampler(
        model,
        dev,
        resolution,
        1;
        num_images=nsamples,
        num_steps=nsteps,
        nspatial=nspatial,
        ntime=ntime,
    )
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt, nspatial=nspatial)
    samples = cpu(samples)

    # loss curves
    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png", nspatial=nspatial)

    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain, samples, savedir, "qq_plot.png", nspatial=nspatial)

    # create plots for comparison of real vs. generated spectra
    spectrum_plot(xtrain, samples, savedir, "mean_spectra.png", nspatial=nspatial)

    # Autocorrelation code 
    # Expects a timeseries of of a scalar: of size nsteps x nbatch
    # Restrict to the first nsamples so that the uncertainties are comparable
    autocorrelation_plot(xtrain[16,16,:,1,:], samples[16,16,:,1,:], savedir, "autocorr.png")

    # Return curve for the following metric: the mean of the middle pixel,
    # taken over a duration time.
    # This may not be using the data fully, since we only extract one "observation" per 
    # sample. However, we are guaranteed that they are independent. 
    duration = 4
    observable(x) = mean(x[16,16,8-div(duration,2):8+div(duration,2)-1,1,:], dims = 1)[:]
    lr = ones(FT, nsamples) # no biasing, likelihood ratio is 1.
    event_probability_plot(observable(xtrain), observable(samples), lr, savedir, "event_probability_$duration.png")

    # make some gifs
    for i in 1:ngifs
        # xtrain    
        anim = @animate for j ∈ 1:size(xtrain, nspatial)
            heatmap(xtrain[:,:,j,1,i])
        end
        path = joinpath(savedir, "anim_fps15_train_$i.gif")
        gif(anim, path, fps = 15)

        # samples
        anim = @animate for j ∈ 1:size(samples, nspatial)
            heatmap(samples[:,:,j,1,i])
        end
        path = joinpath(savedir, "anim_fps15_gen_$(i).gif")
        gif(anim, path, fps = 15)
    end 
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
