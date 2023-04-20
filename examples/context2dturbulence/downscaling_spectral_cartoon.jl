using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using LaTeXStrings
using TOML
using StatsPlots

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

function vary_t0(;source_toml="experiments/Experiment_resize_64_dropout_preprocess_041023.toml",
                  target_toml="experiments/Experiment_all_data_centered_dropout_05_vanilla_loss_unet_mean.toml",
                  use_context = false, FT = Float32, wavenumber = 16)
    wavenumber = FT(wavenumber)
    device = Flux.gpu
    tilesize = 512
    context_channels = 1
    noised_channels = 2

    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, wavenumber; device = device, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, wavenumber; device = device,FT=FT)

    # Samples with all three channels
    sampler = "euler"
    end_times = FT.([0.0,0.16,0.33,0.5, 0.67,0.84, 1])
    ntimes = length(end_times)
    target_samples= zeros(FT, (tilesize, tilesize, context_channels+noised_channels, ntimes)) |> device
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
        if use_context
            target_samples[:,:,1:noised_channels,[i]] .= Euler_Maruyama_sampler(reverse_model, init_x_reverse, time_steps_reverse, Δt_reverse;
                                        c=ctarget[:,:,:,[1]], forward = false)
        else
            target_samples[:,:,1:noised_channels,[i]] .= Euler_Maruyama_sampler(reverse_model, init_x_reverse, time_steps_reverse, Δt_reverse;
            c=ctarget[:,:,:,[1]].*FT(0), forward = false)
        end
    end
   
    target_samples[:,:,1:noised_channels,[1]] .= device(xsource[:,:,1:noised_channels,[1]])
    target_samples[:,:,[3],:] .= ctarget[:,:,:,[1]];
    target_samples = invert_preprocessing(cpu(target_samples), scaling_target);
    plot_array = []
    for t in end_times
        push!(plot_array, Plots.plot(title = L"\mathrm{t}^\star=%$t", border = :none, ticks =nothing,size = (400,50), titlefontsize = 28,plot_titlefontfamily = "TeX Gyre Heros"))
    end
    schemes = (:blues, :bluesreds)
    for ch in 1:noised_channels
        clims = (minimum(target_samples[:,:,ch,:]), maximum(target_samples[:,:,ch,:]))
        for i in 1:ntimes
            t = end_times[i]
            plt = Plots.plot()
            Plots.heatmap!(plt, target_samples[:,:,ch,i],clims = clims,size = (400,400),border = :box, c = schemes[ch])
            push!(plot_array,plt)
        end
    end
    heights = [0.04, 0.48,0.48]
    Plots.plot(plot_array..., layout= grid(3,7, heights = heights), size = (2800,870), colorbar = :none, ticks= nothing)
    Plots.savefig("./vary_t0.png")

    # Since we have the data loaded, make the spectrum plot for vorticity
    source_spectra, k = batch_spectra((xsource |> cpu)[:,:,:,1:20])
    target_spectra, k = batch_spectra((xtarget |> cpu)[:,:,:,1:20])
    values = [0.01,0.2, 3, 50]
    times = [0,0.25,0.5,0.75]
    ch = 2
    plt = Plots.plot(size = (1150,720))
    Plots.plot!(plt,log2.(k), source_spectra[:,ch,1], label = "Low res",linewidth=3, color = "green",plot_titlefontfamily = "TeX Gyre Heros")
    Plots.plot!(plt, log2.(k), target_spectra[:,ch,1], label = "High res",linewidth=3, color = "orange",plot_titlefontfamily = "TeX Gyre Heros")
    xpos = 5.8
    ypos = 2*3.1415/512/512 .* values.^2
    for i in 1:length(values)
        v = values[i]
        t = times[i]
        white_noise = randn(512,512);
        noisy_spectra, k = power_spectrum2d(white_noise*v)
        Plots.plot!(plt, log2.(k), noisy_spectra, label = "",linewidth=3, linecolor = :gray)
        Plots.annotate!(plt, xpos, ypos[i], (L"σ(%$t)=%$v",18, :black))
    end
    Plots.plot!(plt,guidefontsize = 25,legendfontsize =20, tickfontsize = 18,margin = 13Plots.mm, legend =:bottomleft, xlabel = "Wavenumber", ylabel = "Power spectral density", yaxis = :log,fontfamily = "TeX Gyre Heros", fontcolor =:black)
    Plots.plot!(plt, xlim = [0,8], xticks = ([0,2,4,6], [L"2^0", L"2^2", L"2^4", L"2^6"]),tickfontcolor = :black)
    Plots.plot!(plt, ylim = [1e-20,1], yticks = ([1e-20,1e-15,1e-10,1e-5,1], [L"10^{-20}", L"10^{-15}", L"10^{-10}", L"10^{-5}", L"10^{0}"]),tickfontcolor = :black)
    Plots.savefig("./psd_white_noise_tracer.png")
end




function diffusion_bridge_with_context(;source_toml="experiments/Experiment_resize_64_dropout_preprocess_041023.toml",
                                       target_toml="experiments/Experiment_all_data_centered_dropout_05_vanilla_loss_unet_mean.toml")
    FT = Float32
    device = Flux.gpu
    nsteps = 125
    tilesize = 512
    context_channels = 1
    noised_channels = 2
    
    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, 0; device = device, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, 0; device = device,FT=FT)
    
    # Determine which `k` the two sets of images begin to differ from each other
    source_spectra, k = batch_spectra((xsource |> cpu)[:,:,:,1:20])
    target_spectra, k = batch_spectra((xtarget |> cpu)[:,:,:,1:20])
    # create a bridge plot for different contexts with the same source
    images = []
    times = []
    wavenumbers = [16,8,4,2]
    idx = [3, 10, 1, 2];
    source_id = 2
    for i in 1:4
        wavenumber = wavenumbers[i]
        id = idx[i]
        if wavenumber ==16
            cutoff_idx = 5
        elseif wavenumber == 8
            cutoff_idx = 5
        elseif wavenumber == 4
            cutoff_idx = 2
        elseif wavenumber == 2
            cutoff_idx = 1
        end
        
        k_cutoff = FT(k[cutoff_idx])
        target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
        source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))
        reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, FT(512), reverse_model.σ_max, reverse_model.σ_min))
        forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, FT(512), reverse_model.σ_max, reverse_model.σ_min))
        
        init_x = xsource[:,:,:,[source_id]]
        reverse_context = ctarget[:,:,:,[id]]
        forward_context = csource[:,:,:,[source_id]]
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
        push!(images, cat(forward_solution.u..., reverse_solution.u[2:end]..., dims = (4)))
        push!(times, cat(forward_solution.t..., reverse_solution.t[2:end]...,dims = 1))
    end
    # We need to scale each one to have the same range. 
    images = cat(images..., dims = (4));
    maxes = maximum(images, dims = (1,2))
    mins = minimum(images, dims = (1,2))
    images = @. (images - mins) / (maxes - mins)
    images = cpu(images)
    times = cat(times..., dims = (1))
    times = floor.(times .* 100)/100
    ntimes = Int64(length(times)/length(idx))

    c = cpu(ctarget)[:,:,:,idx];
    maxes = maximum(c, dims = (1,2))
    mins = minimum(c, dims = (1,2))
    c = @. (c - mins) / (maxes - mins)
    target_images = cpu(xtarget)[:,:,1,idx];
    maxes = maximum(target_images, dims = (1,2))
    mins = minimum(target_images, dims = (1,2))
    target_images = @. (target_images - mins) / (maxes - mins)
    clims = (0,1)
    plot_array = []
    for j in 1:4
        for i in 1:ntimes
            t = times[i+7*(j-1)]
            if i !=4
                push!(plot_array, Plots.plot(title = L"t = %$t", border = :box, ticks =nothing,size = (400,50), titlefontsize = 28,fontfamily = "TeX Gyre Heros"))
            else
                push!(plot_array, Plots.plot(title = L"t^\star = %$t", border = :box, ticks =nothing,size = (400,50), titlefontsize = 28,fontfamily = "TeX Gyre Heros"))
            end
        end
        push!(plot_array, Plots.plot(title = "Context", border = :box, ticks =nothing,size = (400,50), titlefontsize = 28,fontfamily = "TeX Gyre Heros"))
        push!(plot_array, Plots.plot(title = "Data sample", border = :box, ticks =nothing,size = (400,50), titlefontsize = 28,fontfamily = "TeX Gyre Heros"))
        
        for i in 1:ntimes
            t = times[i+7*(j-1)]
            plt = Plots.plot()
            Plots.heatmap!(plt, images[:,:,1,i+7*(j-1)],clims = clims,size = (400,400), c = :blues)
            push!(plot_array,plt)
        end
        push!(plot_array, Plots.heatmap(c[:,:,1,j], clims = clims, size = (400,400)))
        push!(plot_array, Plots.heatmap(target_images[:,:,j], clims = clims, size = (400,400), c = :blues))
    end
    
    heights = [0.02, 0.23,0.02, 0.23, 0.02, 0.23,0.02, 0.23]
    Plots.plot(plot_array..., layout= grid(8,9, heights = heights), size = (3600,1700), colorbar = :none, border = :none, ticks= nothing)
    Plots.savefig("./downscale.png")
end

function L2plot(; stats_savedir = "stats/512x512/downscale_gen")
    plot1 =Plots.plot(size = (640,400), grid = false)
    plot2 =Plots.plot(size = (640,400), grid = false)
    plots = [plot1, plot2]
    titles = ["Supersaturation", "Vorticity"]
    for ch in 1:2
        for wavenumber in [2.0,4.0,8.0, 16.0]
            L2_filenames = [joinpath(stats_savedir, "downscale_gen_L2_ch1_$wavenumber.csv"),joinpath(stats_savedir, "downscale_gen_L2_ch2_$wavenumber.csv")]
            L2_data = readdlm(L2_filenames[ch], ',')
            int_wavenumber = Int64(wavenumber)
            if wavenumber == 2.0 && ch ==1
                boxplot!(plots[ch],["$int_wavenumber" "$int_wavenumber"],L2_data,outliers = false, whisker_width = :half, fillalpha=0.5, linewidth=2,linecolor=[:purple :green], color=[:purple  :green], label=["True lo-res" "Random lo-res"], fontfamily = "TeX Gyre Heros",fontsize = 28)
            else
                boxplot!(plots[ch],["$int_wavenumber" "$int_wavenumber"],L2_data,outliers = false, whisker_width = :half, fillalpha=0.5, linewidth=2,linecolor=[:purple :green], label = "", color=[:purple  :green],fontfamily = "TeX Gyre Heros",fontsize = 28)
            end
            Plots.plot!(plots[ch],title = titles[ch],fontfamily = "TeX Gyre Heros", fontsize = 28)
            Plots.plot!(plots[ch], ylim = [0,8], xlabel = string("Wavenumber ", L"k_x=k_y"),fontfamily = "TeX Gyre Heros",fontsize = 28)
            if ch == 1
                Plots.plot!(plots[ch], leftmargin = 10Plots.mm, bottom_margin = 5Plots.mm, ylabel = "L² of hi-res and lo-res",fontfamily = "TeX Gyre Heros",fontsize = 28)
            else
                Plots.plot!(plots[ch], leftmargin = 0Plots.mm)
            end
        end
    end
    Plots.plot(plot1,plot2)
    Plots.savefig("L2_conditional.png")
end
