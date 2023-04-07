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
    fraction::FT = 0.05#params.data.fraction
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

function vary_t0(wavenumber;
                 source_toml="experiments/Experiment_resize_64_031422.toml",
                 target_toml="experiments/Experiment_preprocess_mixed_shorter_run_all_data.toml")
    FT = Float32
    device = Flux.gpu
    tilesize = 512
    context_channels = 1
    noised_channels = 2
    ntimes = 11

    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, wavenumber; device = device, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, wavenumber; device = device,FT=FT)

    # Samples with all three channels
    target_samples= zeros(FT, (tilesize, tilesize, context_channels+noised_channels, ntimes)) |> device
    sampler = "euler"
    end_times = FT.(0.0:0.1:1)
    for i in 2:ntimes
        t_end = end_times[i]
        @info t_end
        nsteps = Int64.(round(t_end*500))
        # Set up timesteps for both forward and reverse
        t_forward = zeros(FT, 1) .+ t_end |> device
        time_steps_forward = LinRange(FT(1.0f-5),FT(t_end), nsteps)
        Δt_forward = time_steps_forward[2] - time_steps_forward[1]

        t_reverse = zeros(FT, 1) .+ t_end |> device
        time_steps_reverse = LinRange(FT(t_end), FT(1.0f-5), nsteps)
        Δt_reverse = time_steps_reverse[1] - time_steps_reverse[2]
        init_x_reverse = Euler_Maruyama_sampler(forward_model, xsource[:,:,1:noised_channels,[1]], time_steps_forward, Δt_forward;
                                                c=csource[:,:,:,[1]], forward = true)

        target_samples[:,:,1:noised_channels,[i]] .= Euler_Maruyama_sampler(reverse_model, init_x_reverse, time_steps_reverse, Δt_reverse;
                                    c=ctarget[:,:,:,[1]], forward = false)
    end
   
    target_samples[:,:,1:noised_channels,[1]] .= device(xsource[:,:,1:noised_channels,[1]])
    target_samples[:,:,[3],:] .= ctarget[:,:,:,[1]];
    target_samples = invert_preprocessing(cpu(target_samples), scaling_target);
    plot_array = []
    for t in end_times
        push!(plot_array, Plots.plot(title = "t⋆= $t", border = :none, ticks =nothing,size = (400,50), titlefontsize = 32))
    end

    for ch in 1:noised_channels
        clims = (minimum(target_samples[:,:,ch,:]), maximum(target_samples[:,:,ch,:]))
        for i in 1:ntimes
            t = end_times[i]
            plt = Plots.plot()
            Plots.heatmap!(plt, target_samples[:,:,ch,i],clims = clims,size = (400,400))
            push!(plot_array,plt)
        end
    end
    heights = [0.03, 0.485,0.485]
    Plots.plot(plot_array..., layout= grid(3,11, heights = heights), size = (4400,850), colorbar = :none, border = :none, ticks= nothing)
    Plots.savefig("./vary_t0.png")

    source_spectra, k = new_batch_spectra((xsource |> cpu)[:,:,:,1:20])
    target_spectra, k = new_batch_spectra((xtarget |> cpu)[:,:,:,1:20])
    values = [0.01,0.2, 3, 50, 1000]
    times = [0,0.25,0.5,0.75,1]
    names = ["Moisture", "Vorticity"]
    plot_array = []
    for ch in 1:noised_channels
        plt = Plots.plot()
        Plots.plot!(plt,k, source_spectra[:,ch,1], label = "Low-res",linewidth=3)
        Plots.plot!(plt, k, target_spectra[:,ch,1], label = "High-res",linewidth=3)
        for i in 1:length(values)
            v = values[i]
            t = times[i]
            white_noise = randn(512,512);
            noisy_spectra, k = new_power_spectrum2d(white_noise*v)
            Plots.plot!(plt, k, noisy_spectra, label = "σ($t)=$v",linewidth=3)
        end
        Plots.plot!(plt, margins = 20Plots.mm, guidefontsize=12, titlefontsize = 24, tickfontsize = 8, legend =:topright, xlabel = "Wavenumber", ylabel = "Power Spectral Density", yaxis = :log, title = names[ch])
        push!(plot_array, plt)
    end
    Plots.plot(plot_array..., layout = (1,2), size = (1400,700))
    Plots.savefig("./psd_white_noise.png")
end




function two_model_bridge(;
    source_toml="experiments/Experiment_resize_64_031422.toml",
    target_toml="experiments/Experiment_preprocess_mixed_shorter_run_all_data.toml")
FT = Float32
device = Flux.gpu
nsteps = 125
savepath = "./output"
tilesize = 512
context_channels = 1
noised_channels = 2

forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, 0; device = device, FT=FT)
reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, 0; device = device,FT=FT)

# Determine which `k` the two sets of images begin to differ from each other
source_spectra, k = new_batch_spectra((xsource |> cpu)[:,:,:,1:10])
target_spectra, k = new_batch_spectra((xtarget |> cpu)[:,:,:,1:10])
cutoff_idx = 3
if cutoff_idx > 0
    k_cutoff = FT(k[cutoff_idx])
    target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))

    reverse_t_end = FT(new_t_cutoff(target_power_at_cutoff, k_cutoff, FT(512), reverse_model.σ_max, reverse_model.σ_min))
    forward_t_end = FT(new_t_cutoff(source_power_at_cutoff, k_cutoff, FT(512), reverse_model.σ_max, reverse_model.σ_min))

else
    reverse_t_end = FT(1)
    forward_t_end = FT(1)
end


# create a bridge plot for a two different contexts with the same source
idx = [10,55]
init_x = cat(xsource[:,:,:,[1]],xsource[:,:,:,[1]], dims = 4)
reverse_context = ctarget[:,:,:,idx]
forward_context = cat(csource[:,:,:,[1]],csource[:,:,:,[1]], dims = 4)
forward_solution, reverse_solution = diffusion_bridge_simulation(forward_model,
                                               reverse_model,
                                               init_x,
                                               nsteps;
                                               forward_c=forward_context,
                                               reverse_c=reverse_context,
                                               FT = FT,
                                               forward_t_end=forward_t_end,
                                               forward_solver=DifferentialEquations.EM(),
                                               forward_sde=true,
                                               reverse_solver=DifferentialEquations.EM(),
                                               reverse_t_end=reverse_t_end,
                                               reverse_sde=true
                                               )
# We need to scale each one to have the same range. 
images = cat(forward_solution.u..., reverse_solution.u[2:end]..., dims = (4));
maxes = maximum(images, dims = (1,2))
mins = minimum(images, dims = (1,2))
images = @. (images - mins) / (maxes - mins)
images = cpu(images)
#times = cat(forward_solution.t..., reverse_solution.t[2:end]...,dims = 1);
times = [0,0.06,0.24,0.54,0.24,0.6,0]
ntimes = length(times)
plot_array = []
    for t in times
        push!(plot_array, Plots.plot(title = "t = $t", border = :none, ticks =nothing,size = (400,50), titlefontsize = 32))
    end
    clims = (minimum(images[:,:,1,:]), maximum(images[:,:,1,:]))
    for i in 1:ntimes
        t = times[i]
        plt = Plots.plot()
        Plots.heatmap!(plt, images[:,:,1,2*(i-1)+1],clims = clims,size = (400,400))
        push!(plot_array,plt)
    end
    for i in 1:ntimes
        t = times[i]
        plt = Plots.plot()
        Plots.heatmap!(plt, images[:,:,1,i*2],clims = clims,size = (400,400))
        push!(plot_array,plt)
    end

    heights = [0.03, 0.485,0.485]
    Plots.plot(plot_array..., layout= grid(3,7, heights = heights), size = (2800,850), colorbar = :none, border = :none, ticks= nothing)
    Plots.savefig("./downscale2.png")

end



function main(wavenumber;
              source_toml="experiments/Experiment_resize_64_031422.toml",
              target_toml="experiments/Experiment_preprocess_mixed_shorter_run_all_data.toml")
    FT = Float32
    device = Flux.gpu
    #savedir = "./downscaling_runs/resize_64_031422_preprocess_mixed_shorter_run_all_data"
    nsamples = 5
    tilesize = 512
    context_channels = 1
    noised_channels = 2

    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, wavenumber; device = device, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, wavenumber; device = device,FT=FT)

    # Determine which `k` the two sets of images begin to differ from each other
    #invert_preprocessing(cpu(target_samples), scaling_target)
    source_spectra, k = new_batch_spectra((xsource |> cpu)[:,:,:,1:10])
    target_spectra, k = new_batch_spectra((xtarget |> cpu)[:,:,:,1:10])
    cutoff_idx = min(3,Int64(floor(sqrt(2)*wavenumber-1)))
    if cutoff_idx > 0
        k_cutoff = FT(k[cutoff_idx])
    	target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    	reverse_t_end = FT(new_t_cutoff(target_power_at_cutoff, k_cutoff, FT(512), reverse_model.σ_max, reverse_model.σ_min))
    else
	reverse_t_end = FT(1)
    end

    # Samples with all three channels
    target_samples= zeros(FT, (tilesize, tilesize, context_channels+noised_channels, nsamples)) |> device
    sample_pixels = reshape(target_samples[:,:, 1:noised_channels, :], (prod(size(target_samples)[1:2]), noised_channels, nsamples))

    source_samples = zeros(FT, (tilesize, tilesize, context_channels+noised_channels, nsamples)) |> device

    # This only has the noised channels; these are the IC for the reverse process
    init_x_reverse =  zeros(FT, (tilesize, tilesize, noised_channels, nsamples)) |> device
    sampler = "euler"
    filenames = ["./downscale_cartoon_ch1_$wavenumber.csv",
                 "./downscale_cartoon_ch2_$wavenumber.csv"]

    indices = 1:1:size(csource)[end]
    end_times = FT.(0.05:0.05:1)
    for t_end in end_times
        @info t_end
        nsteps = Int64.(round(t_end*500))
       # Set up timesteps for both forward and reverse
       t_forward = zeros(FT, nsamples) .+ t_end |> device
       time_steps_forward = LinRange(FT(1.0f-5),FT(t_end), nsteps)
       Δt_forward = time_steps_forward[2] - time_steps_forward[1]

       t_reverse = zeros(FT, nsamples) .+ t_end |> device
       time_steps_reverse = LinRange(FT(t_end), FT(1.0f-5), nsteps)
       Δt_reverse = time_steps_reverse[1] - time_steps_reverse[2]



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
        

        target_samples[:,:,[3],:] .= ctarget[:,:,:,1:nsamples];
        sample_spectra = mapslices(x -> hcat(new_power_spectrum2d(x)[1]), cpu(target_samples), dims =[1,2])
        #save the metrics
        for ch in 1:noised_channels
            faithful = sum(Statistics.mean(sample_spectra[1:cutoff_idx,1,ch,:], dims = 2) .- source_spectra[1:cutoff_idx,ch,:][:])
            realistic = sum(Statistics.mean(sample_spectra[(cutoff_idx+1):length(k),1,ch,:], dims = 2) .- target_spectra[(cutoff_idx+1):length(k),ch,:][:])
            output = hcat(t_end, faithful, realistic)
            open(filenames[ch], "a") do io
                writedlm(io, output, ',')
            end
        end
    end
    for ch in 1:noised_channels
#        Plots.plot(k, sample_spectra[:,1,ch,1][:], label = "Samples")
#        Plots.plot!(margin = 20Plots.mm, xlabel = "Wavenumber", ylabel = "Log10(Normalized metric)")
#        Plots.plot!(k, source_spectra[:,ch,:][:], label = "Source")
#        Plots.plot!(k, target_spectra[:,ch,:][:], label = "Target")
#        Plots.plot!(yaxis = :log, xaxis = :log)
#        Plots.plot!(legend = :bottomleft)
#        Plots.savefig("spectra_$ch.png")
        data = readdlm(filenames[ch], ',')
        Plots.plot(data[:,1], log10.(data[:,2]./maximum(data[:,2])), label = "faithful")
        Plots.plot!(margin = 20Plots.mm, xlabel = "Time", ylabel = "Log10(Normalized metric)")
        Plots.plot!(data[:,1], log10.(data[:,3]./maximum(data[:,3])), label = "realistic")
        Plots.plot!([reverse_t_end,reverse_t_end], log10.([0.1,1]), label = "other method")
        Plots.savefig("tmp_$ch.png")
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(parse(Int64, ARGS[1]), parse(Int64, ARGS[2]), parse(Float32, ARGS[3]))
end
