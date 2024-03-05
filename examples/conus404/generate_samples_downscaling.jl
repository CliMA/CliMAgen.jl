using BSON
using Flux
using CUDA
using cuDNN
using HDF5
using JLD2
using ProgressMeter
using Random
using Statistics
using TOML

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/conus404/preprocessing_utils.jl"))
include(joinpath(package_dir,"examples/utils_data.jl")) # for data loading

function setup_sampler_downscaling(x0, tf, model::CliMAgen.AbstractDiffusionModel,
                                    device,
                                    tilesize,
                                    noised_channels;
                                    num_images = 5,
                                    num_steps=500,
                                    ϵ=1.0f-3,
                                    FT=Float32,
                                    )
    t = tf .* ones(FT, num_images) |> device
    num_steps = Int(ceil(num_steps*tf ))
    init_z = randn(FT, (tilesize, tilesize, noised_channels, num_images)) |> device
    _, σ_T = CliMAgen.marginal_prob(model, zero(init_z), t)
    init_noise = (σ_T .* init_z)
    time_steps = LinRange(tf, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_noise .+ x0
end

function generate_samples_downscaling(params; FT=Float32)
    # unpack params, including preprocessing numbers
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    nsamples = params.downscaling.nsamples
    standard_scaling = params.data.standard_scaling
    fname_train = params.data.fname_train
    fname_test = params.data.fname_test
    precip_channel = params.data.precip_channel
    precip_floor::FT = params.data.precip_floor
    n_pixels = params.data.n_pixels
    inchannels = params.model.inchannels
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    # we always train with the preprocessing parameters derived from the
    # training data.
    preprocess_params_file_train = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_train.jld2")
    preprocess_params_file_test = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_test.jld2")

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

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model_smooth)
    
    # sample from the trained model
    xtrain, xtest = get_raw_data_conus404(fname_train, fname_test, precip_channel; precip_floor = precip_floor, FT=FT)
    xtrain_lores = lowpass_filter(xtrain, 8) # guess
    xtest_lores = lowpass_filter(xtest, 8) # guess

    scaling_train = JLD2.load_object(preprocess_params_file_train)
    scaling_test = JLD2.load_object(preprocess_params_file_test)

    xtrain_pp_lores = apply_preprocessing(xtrain_lores, scaling_train)
    xtest_pp_lores = apply_preprocessing(xtest_lores, scaling_test)

    idx = Int.(ceil.(rand(nsamples)*size(xtrain)[end]))
    random_samples  = zeros(FT, (n_pixels, n_pixels, inchannels,nsamples))
    samples_train = zeros(FT, (n_pixels, n_pixels, inchannels,nsamples))
    samples_test = zeros(FT, (n_pixels, n_pixels, inchannels,nsamples))
    tf = 0.6f0 # guess 
    time_steps, Δt, init_x = setup_sampler_downscaling(device(xtrain_pp_lores[:,:,:,idx]), tf,
            model,
            device,
            n_pixels,
            inchannels;
            num_images=nsamples,
            num_steps=nsteps,
        )
    samples_train .= cpu(Euler_Maruyama_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(13)))
    time_steps, Δt, init_x = setup_sampler_downscaling(device(xtest_pp_lores[:,:,:,idx]), tf,
            model,
            device,
            n_pixels,
            inchannels;
            num_images=nsamples,
            num_steps=nsteps,
        )
    samples_test .= cpu(Euler_Maruyama_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(123)))
    # Compute random samples for comparison
    time_steps, Δt, init_x = setup_sampler(
        model,
        device,
        n_pixels,
        inchannels;
        num_images=nsamples,
        num_steps=nsteps,
    )
    random_samples .= cpu(Euler_Maruyama_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(2)))

    samplesdir = joinpath(savedir, "downscaling")
    !ispath(samplesdir) && mkpath(samplesdir)

    samples_file = "samples_downscaled_smooth.hdf5"
    !ispath(samplesdir) && mkpath(samplesdir)
    hdf5_path=joinpath(samplesdir, samples_file)
    fid = HDF5.h5open(hdf5_path, "w")
    fid["downscaled_samples_train"] = invert_preprocessing(samples_train, scaling_train)
    fid["downscaled_samples_test"] = invert_preprocessing(samples_test, scaling_test)
    fid["random_samples_train"] = invert_preprocessing(random_samples, scaling_train)
    fid["random_samples_test"] = invert_preprocessing(random_samples, scaling_test)
    fid["data_train"] = xtrain[:,:,:,idx]
    fid["data_test"] = xtest[:,:,:,idx]
    fid["data_train_lores"] = xtrain_lores[:,:,:,idx]
    fid["data_test_lores"] = xtest_lores[:,:,:,idx]
    close(fid)
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    generate_samples_downscaling(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(;experiment_toml=ARGS[1])
end
