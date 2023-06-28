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

function run_analysis(params; FT=Float32, logger=nothing)
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
        pairs_per_τ = :all,
        f = 0.2,
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
    #samples_self_generating = CliMAgen.Euler_Maruyama_timeseries_sampler(model, init_x, time_steps, Δt; c = xtest[:,:,sample_channels+1:end, :])

    #samples = cpu(samples)

    #clims= (percentile(samples[:],0.1), percentile(samples[:], 99.9))
    #anim = convert_to_animation(samples, 1, clims)
    #gif(anim, string("anim_samples.gif"), fps = 10)

    dt_save = 1.0
    ac, lag, npairs = autocorr(samples, 1, 16, 16)
    ac_l = autocorr_inverse_cdf.(0.05, npairs, ac)
    ac_up = autocorr_inverse_cdf.(0.95, npairs, ac)
    Plots.plot(lag*dt_save, ac,  ribbon = (ac .- ac_l, ac_up .- ac), label = "Generated", ylabel = "Autocorrelation Coeff", xlabel = "Lag (time)", margin = 10Plots.mm)

    #ac_sg, ac_sg_σ, lag = autocorr(samples_self_generating, 1)
    ac_truth, lag = autocorr(xtest, 1, 16, 16) # this is a portion of the timeseries if stride = 1
    ac_truth_l = autocorr_inverse_cdf.(0.05, npairs, ac_truth)
    ac_truth_up = autocorr_inverse_cdf.(0.95, npairs, ac_truth)
    #Plots.plot!(lag*dt_save, ac_sg, ribbon = ac_sg_σ, label = "Generated (Self Generating)")
    Plots.plot!(lag*dt_save, ac_truth, ribbon = (ac_truth .- ac_truth_l, ac_truth_up .- ac_truth), label = "Training")
    Plots.savefig(joinpath(savedir, "autocorr_samples_$nsamples.png"))
# create plot showing distribution of spatial mean of generated and real images
spatial_mean_plot(xtest[:, :, [1], 1:nsamples], samples, savedir, "spatial_mean_distribution.png", logger=logger)

# create q-q plot for cumulants of pre-specified scalar statistics
qq_plot(xtest[:, :, [1], 1:nsamples], samples, savedir, "qq_plot.png", logger=logger)

# create plots for comparison of real vs. generated spectra
spectrum_plot(xtest[:, :, [1], 1:nsamples], samples, savedir, "mean_spectra.png", logger=logger)

# create plots with nimages images of sampled data and testing data
# Rescale now using mintest and maxtest
xtest = @. (xtest - mintest) / (maxtest - mintest)
samples = @. (samples - mintest) / (maxtest - mintest)

img_plot(samples[:, :, [1], 1:nimages], savedir, "$(sampler)_images_ch1.png")
img_plot(xtest[:, :, [1], 1:nimages], savedir, "test_images_ch1.png")
loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    


end

function main(; experiment_toml="Experiment_all_data_dropout.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    run_analysis(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
