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
include(joinpath(package_dir,"examples/utils_etl.jl"))

function run_analysis(params; FT=Float32, logger=nothing)
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

    make_samples = params.sampling.make_samples
    nsamples = params.sampling.nsamples
    nimages = params.sampling.nimages
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    k_bias::FT = params.sampling.k_bias
    shift = params.sampling.shift

    # directory for saving biased samples and plots
    samples_savedir = joinpath(savedir, "biased_$(k_bias)")
    !ispath(samples_savedir) && mkpath(samples_savedir)
    samples_file = joinpath(samples_savedir, params.sampling.samples_file)
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

    # set up dataset
    dl,_ = get_data_correlated_ou1d(batchsize,preprocess_params_file;
                                    f=fraction,
                                    n_pixels=n_pixels,
                                    n_time=n_time,
                                    standard_scaling=standard_scaling,
                                    read =true,
                                    save=false,
                                    FT=FT
                                    )
    xtrain = cat([x for x in dl]..., dims=4)
    xtrain = device(xtrain)
    
    # set up bias for space-time mean
    indicator = zeros(FT, n_pixels, n_time, inchannels)
    indicator[div(n_pixels, 4):3*div(n_pixels, 4), div(n_time, 4):3*div(n_time, 4), :] .= 1
    indicator = device(indicator)

    A(x; indicator = indicator) = sum(indicator .* x, dims=(1, 2, 3)) ./ sum(indicator, dims=(1, 2, 3))
    ∂A∂x(x; indicator = indicator) = indicator ./ sum(indicator, dims=(1, 2, 3))
    bias(x, k = k_bias) = k*∂A∂x(x)

    # Compute normalization using all of the data
    Z = mean(exp.(k_bias .* A(xtrain)))

    if make_samples
        # set up model
        checkpoint_path = joinpath(savedir, "checkpoint.bson")
        BSON.@load checkpoint_path model model_smooth opt opt_smooth
        model = device(model)
        
        # sample from the trained model
        time_steps, Δt, init_x = setup_sampler(
            model,
            device,
            n_pixels,
            inchannels;
            num_images=nsamples,
            num_steps=nsteps,
        )
        if sampler == "euler"
            samples = Euler_Maruyama_ld_sampler(model, init_x, time_steps, Δt, bias=bias, use_shift = shift)
        elseif sampler == "pc"
            error("invalid sampler $sampler.")
        end
        samples = cpu(samples)
    else
        @info samples_file
        samples = read_from_hdf5(samples_file)
    end
    # # Probability of event for the following metric: the mean of the middle pixel,
    # # taken over a time duration length 8
    # # [recall that the image x is of size (n_spatialn_time_steps)]
    duration = 8
    observable(x) = mean(x[32,32-div(duration,2):32+div(duration,2)-1,1,:], dims = 1)[:]
    likelihood_ratio(x; k = k_bias) = Z.*exp.(-k .*A(x; indicator = cpu(indicator)))
    event_probability_plot(observable(cpu(xtrain)), observable(samples), likelihood_ratio(samples)[:], samples_savedir, "event_probability_$(duration)_ld_$(k_bias)_shift_$(shift).png"; logger=logger)

    # # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(cpu(xtrain), samples, samples_savedir, "spatial_mean_distribution_ld.png", logger=logger)

    heatmap_grid(samples[:, :, [1], 1:nimages], 1, samples_savedir, "$(sampler)_images_ld.png")
    heatmap_grid(cpu(xtrain)[:, :, [1], 1:nimages], 1, samples_savedir, "train_images_ld.png")
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
