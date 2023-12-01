using BSON
using Flux
using CUDA
using cuDNN
using HDF5
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML

using CliMAgen
include("../utils_etl.jl")
function generate_samples(params; FT=Float32, bias_sampling = false)
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


    # If we are using a biased sampler, set up the bias function
    # and make a new directory to save in
    if bias_sampling
        k_bias::FT = params.sampling.k_bias
        shift = params.sampling.shift

            # set up bias for space-time mean
        indicator = zeros(FT, n_pixels, n_time, inchannels)
        indicator[div(n_pixels, 4):3*div(n_pixels, 4), div(n_time, 4):3*div(n_time, 4), :] .= 1
        indicator = device(indicator)

        A(x; indicator = indicator) = sum(indicator .* x, dims=(1, 2, 3)) ./ sum(indicator, dims=(1, 2, 3))
        ∂A∂x(x; indicator = indicator) = indicator ./ sum(indicator, dims=(1, 2, 3))
        bias(x, k = k_bias) = k*∂A∂x(x)
    else
        k_bias = params.sampling.k_bias
        bias = nothing
        shift = false
    end

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)

    # sample from the trained model
    samples_per_batch = batchsize
    nbatch = div(nsamples, samples_per_batch)
    all_samples = zeros(FT, (n_pixels, n_time, inchannels,nbatch*samples_per_batch))
    samples = zeros(FT, (n_pixels, n_time, inchannels,samples_per_batch)) |> device
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
            samples .= Euler_Maruyama_ld_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(b), bias=bias, use_shift = shift)
        elseif sampler == "pc" && !bias_sampling
            samples .= predictor_corrector_sampler(model, init_x, time_steps, Δt)
        elseif sampler == "pc" && bias_sampling
            @error("Biased sampling with a Predictor-Corrector sampler is not supported.")
        end
        all_samples[:,:,:,(b-1)*samples_per_batch+1:b*samples_per_batch] .= cpu(samples)
    end
    samplesdir = joinpath(savedir, "bias_$(FT(k_bias))_shift_$shift")
    !ispath(samplesdir) && mkpath(samplesdir)

    drop_to_hdf5(all_samples; hdf5_path=joinpath(samplesdir, samples_file), key = "generated_samples")
end

function main(; experiment_toml="Experiment.toml", bias_sampling = false)
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    generate_samples(params; FT=FT, bias_sampling = bias_sampling)

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(;experiment_toml=ARGS[1], bias_sampling = ARGS[2]== "true")
end
