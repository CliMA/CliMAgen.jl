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

function generate_samples(params; FT=Float32, real_space = true)
    # unpack params, including preprocessing numbers
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    batchsize = params.data.batchsize
    n_pixels = params.data.n_pixels
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    scaling = JLD2.load_object(preprocess_params_file)
    inchannels = params.model.inchannels
    nsamples = params.sampling.nsamples_generate
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    samples_file = params.sampling.samples_file
    
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
    model = device(model)
    
    # sample from the trained model
    # first allocate memory to hold the samples
    samples_per_batch = batchsize
    nbatch = div(nsamples, samples_per_batch)
    samples = zeros(FT, (n_pixels, n_pixels, inchannels,nbatch*samples_per_batch)) # on CPU
    batch = zeros(FT, (n_pixels, n_pixels, inchannels,samples_per_batch)) |> device # on GPU
    for b in 1:nbatch
        time_steps, Δt, init_x = setup_sampler(
            model,
            device,
            n_pixels,
            inchannels;
            num_images=nsamples,
            num_steps=nsteps,
        )
        batch .= Euler_Maruyama_ld_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(b))
        if real_space
            samples[:,:,:,(b-1)*samples_per_batch+1:b*samples_per_batch] .= cpu(invert_preprocessing(batch, scaling))
        else
            samples[:,:,:,(b-1)*samples_per_batch+1:b*samples_per_batch] .= cpu(batch)
        end
        
    end
    samplesdir = savedir
    !ispath(samplesdir) && mkpath(samplesdir)
    hdf5_path=joinpath(samplesdir, samples_file)
    fid = HDF5.h5open(hdf5_path, "w")
    fid["generated_samples"] = samples
    close(fid)
end

function main(; experiment_toml="Experiment.toml", real_space = true)
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    generate_samples(params; FT=FT, real_space = real_space)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(;experiment_toml=ARGS[1], real_space = ARGS[2] == "true")
end
