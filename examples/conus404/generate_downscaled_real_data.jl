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

function generate_downscaled_real_data(params; FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    samplesdir = joinpath(savedir, "downscaling")
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    nsamples = params.downscaling.nsamples
    standard_scaling = params.data.standard_scaling
    fname_train = params.data.fname_train
    fname_test = params.data.fname_test
    precip_channel = params.data.precip_channel
    precip_floor::FT = params.data.precip_floor
    low_pass = params.data.low_pass
    low_pass_k = params.data.low_pass_k
    n_pixels = params.data.n_pixels
    inchannels = params.model.inchannels
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    downscale_samples_file = params.downscaling.downscale_samples_file
    tf::FT = params.downscaling.turnaround_time 
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
    
    !ispath(samplesdir) && mkpath(samplesdir)

    # sample from the trained model
    if params.downscaling.coarse_res_data_file == "nothing"
        # get train and test data (hi-res data)
        wn = 8
        (_, xtest) = get_raw_data_conus404(fname_train, fname_test, precip_channel; precip_floor = precip_floor, FT=FT)
        xtest_hires = xtest
        # Filter to create lo-res standin
        xtest_lores = lowpass_filter(xtest, wn)
        # Preprocess the coarse resolution data
    else
        # get test data (already low-res!) and make preprocessing parameters
        @info "using real coarse resolution data"
        fid = HDF5.h5open(params.downscaling.coarse_res_data_file, "r")
        xtest_lores = HDF5.read(fid["coarse_res_data"])
        close(fid)
    end
    preprocess_params_file_lores = joinpath(samplesdir, params.downscaling.coarse_res_data_preprocess_file)

    save_preprocessing_params(
        xtest_lores, preprocess_params_file_lores; 
        standard_scaling=standard_scaling,
        low_pass=low_pass,
        low_pass_k=low_pass_k,
        FT=FT,
    )
    scaling_lores = JLD2.load_object(preprocess_params_file_lores)
    xtest_pp_lores = apply_preprocessing(xtest_lores, scaling_lores)
    idx = Int.(ceil.(rand(nsamples)*size(xtest_lores)[end]))
    samples = zeros(FT, (n_pixels, n_pixels, inchannels,nsamples))
    time_steps, Δt, init_x = setup_sampler_downscaling(device(xtest_pp_lores[:,:,:,idx]), tf,
            model,
            device,
            n_pixels,
            inchannels;
            num_images=nsamples,
            num_steps=nsteps,
        )
        @info nsamples
    samples .= cpu(Euler_Maruyama_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(123)))
    hdf5_path=joinpath(samplesdir, downscale_samples_file)
    fid = HDF5.h5open(hdf5_path, "w")
    fid["downscaled_samples"] = invert_preprocessing(samples, scaling_lores)
    fid["data_lores"] = xtest_lores[:,:,:,idx]
    if params.downscaling.high_res_data_file != "nothing" && params.downscaling.coarse_res_data_file != "nothing"
        fid_hi = HDF5.h5open(params.downscaling.high_res_data_file, "r")
        xtest_hires = HDF5.read(fid_hi["data"])
        close(fid_hi)
    end
    fid["data_hires"] = xtest_hires[:,:,:,idx]
    close(fid)
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    generate_downscaled_real_data(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(;experiment_toml=ARGS[1])
end
