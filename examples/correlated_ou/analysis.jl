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

function run_analysis(params; FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    resolution = params.data.resolution
    npairs_per_τ = params.data.npairs_per_tau
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    inchannels = params.model.noised_channels
    nsamples = params.sampling.nsamples
    nimages = params.sampling.nimages
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    tilesize_sampling = params.sampling.tilesize
    sample_channels = params.sampling.sample_channels
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
    dl, dl_test  = get_data_correlated_ou2d(
        batchsize;
        pairs_per_τ = npairs_per_τ,
        f = 0.1,
        resolution=resolution,
        FT=Float32,
        standard_scaling = standard_scaling,
        read = true,
        save = false,
        preprocess_params_file,
        rng=Random.GLOBAL_RNG
    )
    # compute max/min using map reduce, then generate a single batch = nsamples
    # use xtest because it has not been shuffled and the images are in the timeseries order
    xtest = cat([x for x in dl_test]..., dims=4)
    
    # To compare statistics from samples and testing data,
    # cut testing data to length nsamples.
    nsamples = 100
    xtest = xtest[:, :, :, 1:nsamples]

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    
    # sample from the model
    time_steps, Δt, init_x = setup_sampler(
        model,
        device,
        tilesize_sampling,
        sample_channels;
        num_images=nsamples,
        num_steps=nsteps,
    )
    # assert sample channels  = half the total channels?

    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c = xtest[:,:,sample_channels+1:end, :] |> device)
    samples_self_generating = CliMAgen.Euler_Maruyama_timeseries_sampler(model, init_x, time_steps, Δt; c = xtest[:,:,sample_channels+1:end, :])

    #samples = cpu(samples)

    #clims= (percentile(samples[:],0.1), percentile(samples[:], 99.9))
    #anim = convert_to_animation(samples, 1, clims)
    #gif(anim, string("anim_samples.gif"), fps = 10)

    dt_save = 1.0
    ac, ac_σ, lag = autocorrelation(samples, 1)
    ac_sg, ac_sg_σ, lag = autocorrelation(samples_self_generating, 1)
    ac_truth, ac_truth_σ, lag = autocorrelation(xtest, 1) # this is a portion of the timeseries if stride = 1
    Plots.plot(lag*dt_save, ac,  ribbon = ac_σ, label = "Generated", ylabel = "Autocorrelation Coeff", xlabel = "Lag (time)", margin = 10Plots.mm)
    Plots.plot!(lag*dt_save, ac_sg, ribbon = ac_sg_σ, label = "Generated (Self Generating)")
    Plots.plot!(lag*dt_save, ac_truth, ribbon = ac_truth_σ, label = "Training")
    Plots.savefig("autocorr_samples_$nsamples.png")



end

function main(; experiment_toml="Experiment_all_data.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    run_analysis(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
