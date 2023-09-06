using BSON
using CUDA
using Flux
using HDF5
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML

using CliMAgen

function read_from_hdf5(params; filename="samples.hdf5")
    savedir = params.experiment.savedir
    
    hdf5_path = joinpath(savedir, filename)
    fid = HDF5.h5open(hdf5_path, "r")

    samples = read(fid["generated_samples"])
    close(fid)

    return samples
end

function drop_to_hdf5(params; FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    batchsize = params.data.batchsize
    n_pixels = params.data.n_pixels
    n_time = params.data.n_time
    @assert n_pixels == n_time
    inchannels = params.model.inchannels
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    nsamples = params.sampling.nsamples
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
    samples_per_batch = batchsize
    nbatch = div(nsamples, samples_per_batch)
    all_samples = zeros(FT, (n_pixels, n_time, inchannels,nbatch*samples_per_batch))
    for b in 1:nbatch
        time_steps, Δt, init_x = setup_sampler(
            model,
            device,
            n_pixels,
            inchannels;
            num_images=samples_per_batch,
            num_steps=nsteps,
        )
        if sampler == "euler"
            samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
        elseif sampler == "pc"
            samples = predictor_corrector_sampler(model, init_x, time_steps, Δt)
        end
        all_samples[:,:,:,(b-1)*samples_per_batch+1:b*samples_per_batch] .= cpu(samples)
    end
    # set up HDF5
    hdf5_path = joinpath(savedir, samples_file)
    fid = HDF5.h5open(hdf5_path, "w")
    fid["generated_samples"] = all_samples
    close(fid)
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    drop_to_hdf5(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
