using BSON
using CUDA
using Flux
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using DelimitedFiles
using StatsBase

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function obtain_context(params, wavenumber, FT)
    # unpack params
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    resolution = params.data.resolution
   # wavenumber::FT = params.data.wavenumber
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels

    # set up dataset - we need this in order to get the context
    dl, _ =  get_data_context2dturbulence(
        batchsize;
        resolution = resolution,
        wavenumber = wavenumber,
        fraction = fraction,
        standard_scaling = standard_scaling,
        FT=FT,
        read=true,
        preprocess_params_file=preprocess_params_file
    )
    
    train = cat([x for x in dl]..., dims=4)
    ctrain = train[:,:,(noised_channels+1):(noised_channels+context_channels),:]
    return ctrain
end

function obtain_model(params)
    savedir = params.experiment.savedir
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    return model
end

function sampling_params(params)
    nsamples = params.sampling.nsamples
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler
    tilesize_sampling = params.sampling.tilesize
    return nsamples, nsteps, sampler, tilesize_sampling
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

function main(nbatches, npixels, wavenumber; experiment_toml="Experiment.toml")
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    rngseed = params.experiment.rngseed
    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    context = obtain_context(params, wavenumber, FT)
    model = obtain_model(params)
    nsamples, nsteps, sampler, tilesize = sampling_params(params)
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels

    savedir = params.experiment.savedir
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

    model = device(model)

    scaling = JLD2.load_object(preprocess_params_file)
    # make these once, here
    samples = zeros(FT, (tilesize, tilesize, context_channels+noised_channels, nsamples)) |> device
    sample_pixels = reshape(samples[:,:, 1:noised_channels, :], (prod(size(samples)[1:2]), noised_channels, nsamples))

    init_x =  zeros(FT, (tilesize, tilesize, noised_channels, nsamples)) |> device
    context = context |> device

    t = ones(FT, nsamples) |> device
    _, σ_T = CliMAgen.marginal_prob(model, init_x, t)
    time_steps = LinRange(FT(1.0), FT(1.0f-5), nsteps)
    Δt = time_steps[1] - time_steps[2]

    indices = 1:1:size(context)[end]
    filenames = [joinpath(savedir, "gen_statistics_ch1_$wavenumber.csv"),joinpath(savedir, "gen_statistics_ch2_$wavenumber.csv")]
    pixel_filenames = [joinpath(savedir, "gen_pixels_ch1_$wavenumber.csv"),joinpath(savedir, "gen_pixels_ch2_$wavenumber.csv")]

    for batch in 1:nbatches
    # Because we do this per wavenumber, all the context values are the same, but
    # we still need an array of size resolutionxresolutionx1xnsamples for sampling
    # If we were doing for all wavenumbers, we'd need a random sample for each batch, below.
        selection = StatsBase.sample(indices, nsamples)
        samples[:,:,1:noised_channels,:] .= generate_samples!(samples[:,:,1:noised_channels,:],init_x,model, context[:,:,:,selection], σ_T, time_steps, Δt, sampler)
        
        # Carry out the inverse preprocessing transform to go back to real space
        samples .= invert_preprocessing(cpu(samples), scaling) |> device

        # compute metrics of interest
        sample_means =  mapslices(Statistics.mean, cpu(samples), dims=[1, 2])
        sample_κ2 = Statistics.var(cpu(samples), dims = (1,2))
        sample_κ3 = mapslices(x -> StatsBase.cumulant(x[:],3), cpu(samples), dims=[1, 2])
        sample_κ4 = mapslices(x -> StatsBase.cumulant(x[:],4), cpu(samples), dims=[1, 2])
        sample_spectra = mapslices(x -> hcat(power_spectrum2d(x)[1]), cpu(samples), dims =[1,2])

        # average instant condensation rate
        sample_icr = make_icr(cpu(samples))

        # samples is 512 x 512 x 3 x 10
        sample_pixels .= reshape(samples[:,:, 1:noised_channels, :], (prod(size(samples)[1:2]), noised_channels, nsamples))
        # We choose the same random selection of pixels for all in nsamples, but this is OK for this dataset.
        pixel_indices = StatsBase.sample(1:1:size(sample_pixels)[1], npixels)

        #save the metrics
        for ch in 1:noised_channels
            # write pixel vaues to other file
            open(pixel_filenames[ch],"a") do io
                writedlm(io, cpu(sample_pixels)[pixel_indices, ch, :], ',')
            end

            if ch == 1
                output = hcat(sample_means[1,1,ch,:],sample_κ2[1,1,ch,:], sample_κ3[1,1,ch,:],sample_κ4[1,1,ch,:], transpose(sample_spectra[:,1,ch,:]), sample_icr[1,1,ch,:])
            else
                output = hcat(sample_means[1,1,ch,:],sample_κ2[1,1,ch,:], sample_κ3[1,1,ch,:],sample_κ4[1,1,ch,:], transpose(sample_spectra[:,1,ch,:]))
            end
            open(filenames[ch], "a") do io
                writedlm(io, output, ',')
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse(Int64, ARGS[1]), parse(Int64, ARGS[2]),  parse(Float32, ARGS[3]); experiment_toml=ARGS[4])
end
