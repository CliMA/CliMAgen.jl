## Script for generating metrics of interest on the generated images ##

using BSON
using Flux
using CUDA
using cuDNN
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using DelimitedFiles
using StatsBase
using HDF5

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function obtain_context(params, wavenumber, FT; nsamples = 25, f = 0.05)
    # unpack params
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    resolution = params.data.resolution
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels
    # set up dataset - we need this in order to get the context
    dl, _ =  get_data_context2dturbulence(
        batchsize;
        resolution = resolution,
        wavenumber = wavenumber,
        fraction = f,
        standard_scaling = standard_scaling,
        FT=FT,
        read=true,
        preprocess_params_file=preprocess_params_file
    )
    
    train = cat([x for x in dl]..., dims=4)
    ctrain = train[:,:,(noised_channels+1):(noised_channels+context_channels),1:nsamples]
    return ctrain
end

function obtain_model(params, ema, epoch)
    savedir = params.experiment.savedir
    checkpoint_path = joinpath(savedir, "checkpoint_$(epoch).bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    if ema 
        return model_smooth
    else
        return model
    end
end

function sampling_params(params)
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    tilesize_sampling = params.sampling.tilesize
    return nsteps, sampler, tilesize_sampling
end

function generate_samples!(samples, init_x, model, context, σ_T, time_steps, Δt, sampler)
    FT = eltype(samples)
    init_x .= randn!(init_x) .* σ_T
    # sample from the trained model
    if sampler == "euler"
        samples .= Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=context)
    elseif sampler == "pc"
        samples .= predictor_corrector_sampler(model, init_x, time_steps, Δt; c=context)
    end
    return samples
end

function main(nbatches; ema = true, bypass = true, epoch = 200.0)
    resolution = 512
    if bypass
        experiment_toml = "experiments/Experiment_single_wavenumber_$(resolution).toml"
    else
        experiment_toml = "experiments/Experiment_single_wavenumber_$(resolution)_mean_bypass_off.toml"
    end
    @info(experiment_toml)
    @assert epoch ∈ [40,80,120,160,200] 
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    rngseed = params.experiment.rngseed
    # set up rng
    rngseed > 0 && Random.seed!(rngseed)
    wavenumber::FT = params.data.wavenumber
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels

    savedir = params.experiment.savedir
    stats_savedir = string("stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)")
    !ispath(stats_savedir) && mkpath(stats_savedir)

    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    nogpu = params.experiment.nogpu
    
    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        @info "Sampling on CPU"
    end
    nsamples = 25
    context = obtain_context(params, wavenumber, FT; f = 0.05, nsamples = nsamples) |> device
    model = obtain_model(params, ema, epoch)
    nsteps, sampler, tilesize = sampling_params(params)
    model = device(model)
    scaling = JLD2.load_object(preprocess_params_file)
    
    # Allocate memory for the generated samples.
    samples = zeros(FT, (tilesize, tilesize, context_channels+noised_channels, nsamples)) |> device

    # Initial condition array
    init_x =  zeros(FT, (tilesize, tilesize, noised_channels, nsamples)) |> device

    # Setup timestepping simulations
    t = ones(FT, nsamples) |> device
    _, σ_T = CliMAgen.marginal_prob(model, init_x, t)
    time_steps = LinRange(FT(1.0), FT(1.0f-5), nsteps)
    Δt = time_steps[1] - time_steps[2]

    # Filenames for output
    filenames = [joinpath(stats_savedir, "statistics_ch1_$wavenumber.csv"),joinpath(stats_savedir, "statistics_ch2_$wavenumber.csv")]
    for batch in 1:nbatches
        @info batch
        # Sample generation only fills in the noised channels; the contextual channels are left untouched.
        samples[:,:,1:noised_channels,:] .= generate_samples!(samples[:,:,1:noised_channels,:],init_x,model, context, σ_T, time_steps, Δt, sampler)
        
        # Carry out the inverse preprocessing transform to go back to real space
        samples .= invert_preprocessing(cpu(samples), scaling) |> device

        # compute metrics of interest
        sample_means =  mapslices(Statistics.mean, cpu(samples), dims=[1, 2])
        sample_κ2 = Statistics.var(cpu(samples), dims = (1,2))
        sample_κ3 = mapslices(x -> StatsBase.cumulant(x[:],3), cpu(samples), dims=[1, 2])
        sample_κ4 = mapslices(x -> StatsBase.cumulant(x[:],4), cpu(samples), dims=[1, 2])
        sample_spectra = mapslices(x -> hcat(power_spectrum2d(x)[1]), cpu(samples), dims =[1,2])
        if batch == 1
            #Save
            fname = joinpath(stats_savedir, "samples.hdf5")
            fid = h5open(fname, "w")
            fid[string("samples")] = cpu(samples)
            close(fid)
        end
        #save the metrics
        for ch in 1:noised_channels
            output = hcat(sample_means[1,1,ch,:],sample_κ2[1,1,ch,:], sample_κ3[1,1,ch,:],sample_κ4[1,1,ch,:], transpose(sample_spectra[:,1,ch,:]))
            open(filenames[ch], "a") do io
                writedlm(io, output, ',')
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse(Int64, ARGS[1]); ema = parse(Bool, ARGS[2]),  bypass = parse(Bool, ARGS[3]), epoch = parse(Float64, ARGS[4]))
end
