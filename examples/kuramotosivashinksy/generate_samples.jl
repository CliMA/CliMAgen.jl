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
include("../utils_data.jl")
function generate_samples(params; FT=Float32, k_bias=0.0f0, n_avg=1)
    # unpack params
    if params.sampling.sampler == "pc"
        @error("Biased sampling with a Predictor-Corrector sampler is not supported")
    end
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

    # set up bias for space-time mean
    indicator = zeros(FT, n_pixels, n_time, inchannels)
    midx = Int(div(n_pixels, 2))
    midy = Int(div(n_time, 2))
    indicator[midx-n_avg+1:midx, midy-n_avg+1:midy, :] .= 1
    indicator = device(indicator)

    A(x; indicator = indicator) = sum(indicator .* x, dims=(1, 2, 3)) ./ sum(indicator, dims=(1, 2, 3))
    ∂A∂x(x; indicator = indicator) = indicator ./ sum(indicator, dims=(1, 2, 3))
    bias(x, k = k_bias) = k*∂A∂x(x)
    shift = params.sampling.shift

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
        samples .= Euler_Maruyama_ld_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(b), bias=bias, use_shift = shift)
        all_samples[:,:,:,(b-1)*samples_per_batch+1:b*samples_per_batch] .= cpu(samples)
    end
    # Compute normalization using all of the data
     dl,_ = get_data_ks(batchsize,preprocess_params_file;
                        f=fraction,
                        n_pixels=n_pixels,
                        n_time=n_time,
                        standard_scaling=standard_scaling,
                        read =true,
                        save=false,
                        FT=FT
                        )
    xtrain = cat([x for x in dl]..., dims=4);
    Z = mean(exp.(k_bias .* A(xtrain;indicator = cpu(indicator)))[:])
    # Compute the likelihood ratio of the samples
    lr = Z.*exp.(-k_bias .*A(all_samples; indicator = cpu(indicator)))[:]
  
    samplesdir = joinpath(savedir, "bias_$(FT(k_bias))_n_avg_$(n_avg)_shift_$shift")
    !ispath(samplesdir) && mkpath(samplesdir)
    hdf5_path=joinpath(samplesdir, samples_file)
    fid = HDF5.h5open(hdf5_path, "w")
    fid["generated_samples"] = all_samples
    fid["likelihood_ratio"] = lr
    close(fid)
end

function main(; experiment_toml="Experiment.toml", k_bias=0.0f0, n_avg=1)
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    generate_samples(params; FT=FT, k_bias=FT(k_bias), n_avg=Int(n_avg))

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(;experiment_toml=ARGS[1], k_bias=parse(Float64, ARGS[2]), n_avg=parse(Int64, ARGS[3]))
end
