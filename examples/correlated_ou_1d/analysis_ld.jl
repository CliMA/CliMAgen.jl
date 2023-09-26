using BSON
using CUDA
using Flux
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
include(joinpath(package_dir,"examples/correlated_ou_1d/store_load_samples.jl"))

function run_analysis(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    samples_savedir = joinpath(savedir, "biased")
    !ispath(samples_savedir) && mkpath(samples_savedir)

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
    samples_file = params.sampling.samples_file
    nsamples = params.sampling.nsamples
    nimages = params.sampling.nimages
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    k_bias::FT = params.sampling.k_bias

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
        if sampler == "euler_ld"
            samples = Euler_Maruyama_ld_sampler(model, init_x, time_steps, Δt, bias=bias)
        elseif sampler == "pc"
            error("invalid sampler $sampler.")
        end
        samples = cpu(samples)
    else
        samples = read_from_hdf5(params, filename=samples_file)
    end
    # # Return curve for the following metric: the mean of the middle pixel,
    # # taken over a block of time length 8
    m = 8
    observable(x) = mean(x[32,32-div(m,2):32+div(m,2)-1,1,:], dims = 1)[:]
    likelihood_ratio(x; k = k_bias) = Z.*exp.(-k .*A(x; indicator = cpu(indicator)))
    event_probability_plot(observable(cpu(xtrain)), observable(samples), likelihood_ratio(samples)[:], samples_savedir, "event_probability_$(m)_ld.png"; logger=logger)

    # # Im not sure about the following: 
    # # To compute the return time, we need more care. We need a time interval associated with this event in 
    # # order to turn a probability into a return time. If the block length is longer than the autocorrelation
    # # time, I think we can use the block length directly: within each sample of length n_time > m, we get one
    # # independent sample of length m. If instead we had carried out a direct numerical simulation
    # # and split it into blocks of length m, we would get the same result because the blocks would be independent.

    # # The issue arises if the block is shorter than the autocorrelation time. In this case, if we 
    # # had carried out a direct numerical simulation and split it into blocks of length m,
    # # the blocks would no longer be independent. Since these do not agree, I dont think it's the
    # # right thing to do.

    # # We could try: min(autocorrelation time, block length), or split each sample into many blocks
    # # of length m and take the maximum. Then even if they are correlated, we return an independent 
    # # sample per n_time.
    # metric_return_time(y) = maximum(mapslices(x -> block_applied_func(x, mean, m), y, dims = 1), dims = 1)[:]
    # return_curve_plot(metric_return_time(xtrain), metric_return_time(samples), FT(n_time), savedir, "return_curve_$(m)_ld.png"; logger=logger)

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
