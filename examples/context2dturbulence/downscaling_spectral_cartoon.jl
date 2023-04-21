using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Random
using Statistics
using LaTeXStrings
using TOML
using StatsPlots
using CairoMakie
using Plots

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function unpack_experiment(experiment_toml, wavenumber; device = Flux.gpu, FT=Float32)
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
     # set up rng
    rngseed = params.experiment.rngseed
    rng = Random.MersenneTwister(rngseed)
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
        rng = rng,
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

function vary_t0(plotname;
                xsource,
                csource,
                xtarget,
                ctarget,
                scaling_target,
                resolution = 512,
                noised_channels = 2,
                context_channels = 1,
                use_context = false,
                FT = Float32,
                device = Flux.gpu)
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

    schemes = (Reverse(:dense), :bluesreds)
    fig = CairoMakie.Figure(resolution = (2800,870), fontsize=35)
    for ch in 1:noised_channels
        clims = (minimum(target_samples[:,:,ch,:]), maximum(target_samples[:,:,ch,:]))
        for i in 1:ntimes
            t = end_times[i]
            ax = CairoMakie.Axis(fig[ch,i], title = L"t^\star=%$t", titlefont = :regular, xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, )
            CairoMakie.heatmap!(ax, target_samples[:,:,ch,i],clims = clims,colormap = schemes[ch],border = :box, )
        end
    end
    CairoMakie.save(plotname, fig)
end

function psd_white_noise(plotname;
                         xsource = xsource,
                         xtarget=xtarget,
                         resolution = 512,
                         channel = 2)
    # Since we have the data loaded, make the spectrum plot for vorticity
    source_spectra, k = batch_spectra((xsource |> cpu)[:,:,:,1:20])
    target_spectra, k = batch_spectra((xtarget |> cpu)[:,:,:,1:20])
    values = [0.01,0.2, 3, 50]
    times = [0,0.25,0.5,0.75]
    fig = CairoMakie.Figure(resolution = (1000,720), fontsize=28)
    ax = CairoMakie.Axis(fig[1,1], ylabel="Power spectral density", xlabel = "Wavenumber", yscale = log10)
    CairoMakie.xlims!(ax, [0, 8])
    CairoMakie.ylims!(ax, [1e-20,1])
    CairoMakie.lines!(ax, log2.(k), source_spectra[:,channel,1], linewidth = 3, color = "green", label= "Low res.", xticks = ([0,2,4,6], [L"2^0", L"2^2", L"2^4", L"2^6"]), yticks = ([1e-20,1e-15,1e-10,1e-5,1], [L"10^{-20}", L"10^{-15}", L"10^{-10}", L"10^{-5}", L"10^{0}"]))
    CairoMakie.lines!(ax, log2.(k), target_spectra[:,channel,1], label = "High res", color = "orange", linewidth = 3)
    xpos = 5.0
    ypos = 1.5*3.1415/resolution/resolution .* values.^2
    for i in 1:length(values)
        v = values[i]
        t = times[i]
        white_noise = randn(resolution,resolution);
        noisy_spectra, k = power_spectrum2d(white_noise*v)
        CairoMakie.lines!(ax, log2.(k), noisy_spectra, label = "",color = :black, linewidth=3)
        CairoMakie.text!(ax, xpos, ypos[i]; text = L"σ(%$t)=%$v", color = :black, fontsize = 28)
    end
    CairoMakie.save(plotname, fig)
end


function diffusion_bridge_with_context(plotname;
                                       forward_model,
                                       reverse_model,
                                       xsource,
                                       xtarget,
                                       csource,
                                       ctarget,
                                       scaling_source,
                                       scaling_target,
                                       device = Flux.gpu,
                                       FT = Float32,
                                       resolution = 512,
                                       noised_channels = 2,
                                       context_channels = 1,
                                       nsteps = 500)
    # Determine which `k` the two sets of images begin to differ from each other
    source_spectra, k = batch_spectra((xsource |> cpu)[:,:,:,1:20])
    target_spectra, k = batch_spectra((xtarget |> cpu)[:,:,:,1:20])
    # create a bridge plot for different contexts with the same source
    images = []
    times = []
    wavenumbers = [16,8,4,2]
    idx = [6, 2, 17, 10];
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
        reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, FT(resolution), reverse_model.σ_max, reverse_model.σ_min))
        forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, FT(resolution), reverse_model.σ_max, reverse_model.σ_min))
        
        init_x = xsource[:,:,:,[source_id]]
        reverse_context = ctarget[:,:,:,[id]]
        forward_context = csource[:,:,:,[source_id]]
        nsteps = Int(round(nsteps*(reverse_t_end+forward_t_end)/2))
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
    fig = CairoMakie.Figure(resolution = (3600,1800), fontsize=48)
    for j in 1:4
        for i in 1:ntimes
            t = times[i+7*(j-1)]
            if i !=4
                ax = CairoMakie.Axis(fig[j,i], title = L"t = %$t", titlefont = :regular, xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, )
            else
                ax = CairoMakie.Axis(fig[j,i], title = L"t^\star = %$t", titlefont = :regular, xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, )
            end
            CairoMakie.heatmap!(ax, images[:,:,1,i+7*(j-1)],clims = clims,colormap = Reverse(:dense),border = :box, )
        end
        ax = CairoMakie.Axis(fig[j,8], title = "Context", titlefont = :regular, xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, )
        CairoMakie.heatmap!(ax, c[:,:,1,j],clims = clims,colormap = :inferno,border = :box, )
        ax = CairoMakie.Axis(fig[j,9], title = "Data sample", titlefont = :regular, xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, )
        CairoMakie.heatmap!(ax, target_images[:,:,j],clims = clims,colormap = Reverse(:dense),border = :box, )
    end
    CairoMakie.save(plotname, fig)
end

function L2plot(plotname; stats_savedir = "stats/512x512/downscale_gen")
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
                StatsPlots.boxplot!(plots[ch],["$int_wavenumber" "$int_wavenumber"],L2_data,outliers = false, whisker_width = :half, fillalpha=0.5, linewidth=2,linecolor=[:purple :green], color=[:purple  :green], label=["True low res." "Random low res."], fontfamily = "TeX Gyre Heros",fontsize = 28)
            else
                StatsPlots.boxplot!(plots[ch],["$int_wavenumber" "$int_wavenumber"],L2_data,outliers = false, whisker_width = :half, fillalpha=0.5, linewidth=2,linecolor=[:purple :green], label = "", color=[:purple  :green],fontfamily = "TeX Gyre Heros",fontsize = 28)
            end
            Plots.plot!(plots[ch],title = titles[ch],fontfamily = "TeX Gyre Heros", fontsize = 28)
            Plots.plot!(plots[ch], ylim = [0,1], xlabel = string("Wavenumber ", L"k_x=k_y"),fontfamily = "TeX Gyre Heros",fontsize = 28)
            if ch == 1
                Plots.plot!(plots[ch], leftmargin = 10Plots.mm, bottom_margin = 5Plots.mm, ylabel = "L² of high res. and low res.",fontfamily = "TeX Gyre Heros",fontsize = 28)
            else
                Plots.plot!(plots[ch], leftmargin = 0Plots.mm)
            end
        end
    end
    Plots.plot(plot1,plot2)
    Plots.savefig(plotname)
end

function superposition_plot(plotname;
                            forward_model,
                            reverse_model,
                            xsource,
                            csource,
                            scaling,
                            device = Flux.gpu,
                            FT = Float32,
                            noised_channels = 2,
                            resolution=512)
    amp = 1.0;
    n = resolution;
    k1 = 16.0;
    k2 = 32.0
    L = 2π
    xx = ones(n) * LinRange(0,L,n)' 
    yy = LinRange(0,L,n) * ones(n)'
    context1 = @. amp * sin(2π * k1 * xx / L) * sin(2π *  k1 * yy / L)
    context2 = @. amp * sin(2π * k2 * xx / L) * sin(2π *  k2 * yy / L)
    context_raw = context1 .* (xx .> π) + context2 .* (xx .<= π);
    context_scaling = CliMAgen.MeanSpatialScaling([scaling.mintrain_mean[3]],
                                                       [scaling.Δ̄[3]],
                                                       [scaling.mintrain_p[3]],
                                                       [scaling.Δp[3]])
    
    context_preprocessed = apply_preprocessing(context_raw, context_scaling) |> device
    context_preprocessed = reshape(context_preprocessed, (resolution,resolution,1,1));
    # Samples with all three channels
    sampler = "euler"
    t_end = FT(0.47)
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

    target_samples = Euler_Maruyama_sampler(reverse_model, init_x_reverse, time_steps_reverse, Δt_reverse;
                                                                        c=context_preprocessed, forward = false)

    target_samples = cat(target_samples, context_preprocessed, dims = (3));
    target_samples = invert_preprocessing(cpu(target_samples), scaling);
    ch = 1
    clims = (minimum(target_samples[:,:,ch,:]), maximum(target_samples[:,:,ch,:]))
    clims_c = (minimum(target_samples[:,:,3,:]), maximum(target_samples[:,:,3,:]))
    spectrum, k = batch_spectra(target_samples[:,:,1,1])

    # fig
    fig = CairoMakie.Figure(resolution = (1000,1000), fontsize=28)

    ax = CairoMakie.Axis(fig[1,1], ylabel="", title="Generated sample", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, titlefont = :regular)
    co = CairoMakie.heatmap!(ax, target_samples[:,:,1,1], colormap=Reverse(:dense), clims = clims,border = :box)

    ax = CairoMakie.Axis(fig[1,2], title="Context", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, titlefont = :regular)
    co = CairoMakie.heatmap!(ax, target_samples[:,:,3,1], colormap=:inferno, clims = clims_c,border = :box)

    ax = CairoMakie.Axis(fig[2,1:2],  title="Spectrum", xlabel = "Wavenumber", ylabel = "Power spectral density", titlefont = :regular,yscale = log10)
    CairoMakie.xlims!(ax, [0, 8])
    CairoMakie.ylims!(ax, [1e-8,1])
    co = CairoMakie.lines!(ax, log2.(k), spectrum[:], xticks = ([0,2,4,6], [L"2^0", L"2^2", L"2^4", L"2^6"]), yticks = ([1e-8,1e-4,1], [ L"10^{-8}", L"10^{-4}", L"10^{0}"]), color = :black)
    #hidedecorations!(ax, ticks = false, label = false, ticklabels = false)
    CairoMakie.lines!(ax,[log2.(sqrt(2)*16), log2.(sqrt(2)*16)], [1e-8, 1], linestyle = :dot, label = "",color = :black)
    CairoMakie.lines!(ax, [log2.(sqrt(2)*32), log2.(sqrt(2)*32)], [1e-8, 1], linestyle = :dot, label = "",color = :black)
    CairoMakie.save(plotname, fig)
end

function main(;source_toml="experiments/Experiment_resize_64_dropout_preprocess_041023.toml",
               target_toml="experiments/Experiment_all_data_centered_dropout_05.toml",
               FT = Float32)
    params = TOML.parsefile(target_toml)
    params = CliMAgen.dict2nt(params)
    resolution = params.data.resolution
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels
    tilesize = resolution

    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, 0; device = Flux.gpu, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, 0; device = Flux.gpu,FT=FT)

    L2plot("./L2_conditional.png"; stats_savedir = "stats/512x512/downscale_gen")

    superposition_plot("./superposition_context.png";
                       forward_model = forward_model,
                       reverse_model = reverse_model,
                       xsource = xsource,
                       csource = csource,
                       scaling = scaling_target)

    diffusion_bridge_with_context("./downscale.png";
                                  forward_model = forward_model,
                                  reverse_model,
                                  xsource = xsource,
                                  xtarget = xtarget,
                                  csource = csource,
                                  ctarget = ctarget,
                                  scaling_source = scaling_source,
                                  scaling_target = scaling_target)

    psd_white_noise("./psd_white_noise_vorticity.png";
                    xsource = xsource,
                    xtarget=xtarget,
                    channel = 2)

    vary_t0("./vary_t0.png";
            xsource,
            csource,
            xtarget,
            ctarget,
            scaling_target,
            resolution = 512,
            noised_channels = 2,
            context_channels = 1,
            use_context = false,
            FT = Float32,
            device = Flux.gpu)
end