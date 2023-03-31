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

function unpack_experiment(experiment_toml, wavenumber; device = Flux.gpu, FT=Float32)
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    # unpack params
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction::FT = params.data.fraction
    standard_scaling  = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    scaling = JLD2.load_object(preprocess_params_file)
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels

    if resolution == 64
        wavenumber = 0
        # Our coarse res data essentially has flat context. If we did have nonzero context fields,
        # we'd have to be careful to get the right wavenumber
        # so that the diffusion bridge is done for the same context in both directions.
    end
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

    xtrain = train[:,:,1:noised_channels,:] |> device
    ctrain = train[:,:,(noised_channels+1):(noised_channels+context_channels),:] |> device

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    return model, xtrain, ctrain, scaling
end

function generate_samples!(samples, init_x, model, context, time_steps, Δt, sampler; forward = false)
    # sample from the trained model
    if sampler == "euler"
        samples .= Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=context, forward=forward)
    elseif sampler == "pc"
        @error("Not yet supported")
    end
    return samples
end

function filter_512_to_64(x)
    FT = eltype(x)
    y = Complex{FT}.(x)
    # Filter.
    fft!(y, (1,2));
    y[:,33:479,:,:] .= Complex{FT}(0);
    y[33:479,:,:,:] .= Complex{FT}(0);
    ifft!(y, (1,2))
    return real(y)
end

function main(nbatches, npixels, wavenumber;
              source_toml="experiments/Experiment_resize_64_031422.toml",
              target_toml="experiments/Experiment_preprocess_mixed_shorter_run_all_data.toml")
    FT = Float32
    device = Flux.gpu
    nsteps = 125
    savedir = "./downscaling_runs/resize_64_031422_preprocess_mixed_shorter_run_all_data"
    nsamples = 10
    tilesize = 512
    context_channels = 1
    noised_channels = 2

    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, wavenumber; device = device, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, wavenumber; device = device,FT=FT)

    # Determine which `k` the two sets of images begin to differ from each other
    source_spectra, k = batch_spectra((xsource |> cpu)[:,:,:,1:10], size(xsource)[1])
    target_spectra, k = batch_spectra((xtarget |> cpu)[:,:,:,1:10], size(xtarget)[1])

    cutoff_idx = Int64(floor(sqrt(2)*wavenumber-1))
    if cutoff_idx > 0
        k_cutoff = FT(k[cutoff_idx])

    	source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))
    	forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, forward_model.σ_max, forward_model.σ_min))
    	target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    	reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, reverse_model.σ_max, reverse_model.σ_min))
    else
	reverse_t_end = FT(1)
	forward_t_end = FT(1)
    end
    forward_t_end = reverse_t_end
    @show forward_t_end
    @show reverse_t_end
    # Samples with all three channels
    target_samples= zeros(FT, (tilesize, tilesize, context_channels+noised_channels, nsamples)) |> device
    sample_pixels = reshape(target_samples[:,:, 1:noised_channels, :], (prod(size(target_samples)[1:2]), noised_channels, nsamples))

    source_samples = zeros(FT, (tilesize, tilesize, context_channels+noised_channels, nsamples)) |> device

    # This only has the noised channels; these are the IC for the reverse process
    init_x_reverse =  zeros(FT, (tilesize, tilesize, noised_channels, nsamples)) |> device

    # Set up timesteps for both forward and reverse
    t_forward = zeros(FT, nsamples) .+ forward_t_end |> device
    time_steps_forward = LinRange(FT(1.0f-5),FT(forward_t_end), nsteps)
    Δt_forward = time_steps_forward[2] - time_steps_forward[1]

    t_reverse = zeros(FT, nsamples) .+ reverse_t_end |> device
    time_steps_reverse = LinRange(FT(reverse_t_end), FT(1.0f-5), nsteps)
    Δt_reverse = time_steps_reverse[1] - time_steps_reverse[2]

    sampler = "euler"
    filenames = [joinpath(savedir, "downscale_gen_statistics_ch1_$wavenumber.csv"),
                 joinpath(savedir, "downscale_gen_statistics_ch2_$wavenumber.csv")]
    pixel_filenames = [joinpath(savedir, "downscale_gen_pixels_ch1_$wavenumber.csv"),joinpath(savedir, "downscale_gen_pixels_ch2_$wavenumber.csv")]

    indices = 1:1:size(csource)[end]

    for batch in 1:nbatches
        selection = StatsBase.sample(indices, nsamples)
        # Integrate forwards to fill init_x_reverse in place
        # The IC for this step are xsource[:,:,1:noised_channels,selection]
        # Context is passed in as a separate field.
        init_x_reverse .= generate_samples!(init_x_reverse,
                                            xsource[:,:,1:noised_channels,selection], #forward IC
                                            forward_model,
                                            csource[:,:,:,selection],# forward context
                                            time_steps_forward,
                                            Δt_forward,
                                            sampler;
                                            forward = true);

        # If the models use different sigma schedules or if the times are different, we need to adjust
        adapt_x!(init_x_reverse, forward_model, reverse_model, forward_t_end, reverse_t_end);
        
        # Integrate backwards to fill the noised channels of target_samples in place.
        # Since we do this by wavenumber, all the target context are the same
        target_samples[:,:,1:noised_channels,:] .= generate_samples!(target_samples[:,:,1:noised_channels,:],
                                                              init_x_reverse,
                                                              reverse_model,
                                                              ctarget[:,:,:,1:nsamples],
                                                              time_steps_reverse,
                                                              Δt_reverse,
                                                              sampler;
                                                              forward = false);
        

        # Carry out the inverse preprocessing transform to go back to real space
        # Since samples is on the GPU, and invert preprocessing for some reason does not run on GPU,
        # do this conversion to cpu and back to GPU. TODO: figure out why.

        # Fill in channel info on samples for inverting the preprocessing
        # Preprocessing acts on both noised and context channels
        target_samples[:,:,[3],:] .= ctarget[:,:,:,1:nsamples];
        target_samples = invert_preprocessing(cpu(target_samples), scaling_target) |> device;

        source_samples[:,:,1:noised_channels,:] .= xsource[:,:,1:noised_channels,selection];
        source_samples[:,:,[3],:] .= csource[:,:,:,selection];
        source_samples = invert_preprocessing(cpu(source_samples), scaling_source) |> device

        # Save nsamples of the initial and final images in real space
        if batch == 1
	    imgfile = joinpath(savedir, "downscaling_images_$wavenumber.jld2")
            jldsave(imgfile; source = cpu(source_samples),
                    target = cpu(target_samples),
                    lo_res_target = filter_512_to_64(cpu(target_samples)))
        end
        # Load with
        # source = load("./downscaling_images.jld2")["source"]; e.g.

        # compute metrics of interest
        sample_means =  mapslices(Statistics.mean, cpu(target_samples), dims=[1, 2])
        sample_κ2 = Statistics.var(cpu(target_samples), dims = (1,2))
        sample_κ3 = mapslices(x -> StatsBase.cumulant(x[:],3), cpu(target_samples), dims=[1, 2])
        sample_κ4 = mapslices(x -> StatsBase.cumulant(x[:],4), cpu(target_samples), dims=[1, 2])
        sample_spectra = mapslices(x -> hcat(power_spectrum2d(x, 512)[1]), cpu(target_samples), dims =[1,2])

        # average instant
        sample_icr = make_icr(cpu(target_samples))

        # samples is 512 x 512 x 3 x 10
        sample_pixels .= reshape(target_samples[:,:, 1:noised_channels, :], (prod(size(target_samples)[1:2]), noised_channels, nsamples))
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
    main(parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Float32, ARGS[3]))
end
