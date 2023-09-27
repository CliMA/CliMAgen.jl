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

function unpack_experiment(experiment_toml; nogpu = false, FT=Float32)
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    rngseed = params.experiment.rngseed
    savedir = params.experiment.savedir

    batchsize = params.data.batchsize
    tilesize = params.data.tilesize
    kernel_std = params.data.kernel_std
    standard_scaling = params.data.standard_scaling
    bias_wn = params.data.bias_wn
    bias_amplitude = params.data.bias_amplitude

    rngseed > 0 && Random.seed!(rngseed)

     # set up device
     if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        @info "Sampling on CPU"
    end

    dl, _ = get_data_2dturbulence(
        batchsize;
        width=(tilesize, tilesize),
        stride=(tilesize, tilesize),
        kernel_std=kernel_std,
        standard_scaling=standard_scaling,
        bias_wn =bias_wn,
        bias_amplitude=bias_amplitude,
        FT=FT
    )
    xtrain = cat([x for x in dl]..., dims=4)

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    return model, xtrain |> device
end


function two_model_bridge(;
                          source_toml="experiments/Experiment256_larger_bias_and_blurry.toml",
                          target_toml="experiments/Experiment256_base_copy.toml")
    FT = Float32
    nogpu = false
    nsteps = 125
    savepath = "./"
    nsamples = 10
    channel = 1

    forward_model, xsource = unpack_experiment(source_toml; nogpu=nogpu, FT=FT)
    reverse_model, xtarget = unpack_experiment(target_toml; nogpu=nogpu, FT=FT)

    # Determine which `k` the two sets of images begin to differ from each other
    
    source_spectra, k = batch_spectra(xsource |> cpu)
    target_spectra, k = batch_spectra(xtarget |> cpu)

    # this is manual right now. just eyeball it.
    cutoff_idx = 3
    k_cutoff = FT(k[cutoff_idx])
    N = FT(size(xsource)[1])
    source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))
    forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, N, forward_model.σ_max, forward_model.σ_min))
    target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, N, reverse_model.σ_max, reverse_model.σ_min))

    # create a bridge plot for a single image.
    init_x = xsource[:,:,:,[3]]
    plotname = "sde_reverse_partial.png"
    forward_solution, reverse_solution = diffusion_bridge_simulation(forward_model,
                                                                     reverse_model,
                                                                     init_x,
                                                                     nsteps;
                                                                     forward_t_end=forward_t_end,
                                                                     forward_solver=DifferentialEquations.RK4(),
                                                                     forward_sde=false,
                                                                     reverse_solver=DifferentialEquations.EM(),
                                                                     reverse_t_end=reverse_t_end,
                                                                     reverse_sde=true
                                                                     )
    # To make the plot with grayscale, we need to scale each one to between 0 and 1. 
    images = cat(forward_solution.u..., reverse_solution.u[2:end]..., dims = (4));
    maxes = maximum(images[:,:,[channel],:], dims = (1,2))
    mins = minimum(images[:,:,[channel],:], dims = (1,2))
    images = @. (images[:,:,[channel],:] - mins) / (maxes - mins)
    img_plot(images, savepath, plotname; ncolumns = size(images)[end])
    
end


function three_model_bridge(;m1_toml="experiments/Experiment256_larger_bias_and_blurry.toml",
                             m2_toml="experiments/Experiment256_blurry_copy.toml",
                             m3_toml="experiments/Experiment256_base_copy.toml")
    FT = Float32
    nogpu = false
    nsteps = 125
    savepath = "./"
    nsamples = 10
    ϵ=1.0f-5
    nimages=4
    channel = 1

    model1, x1 = unpack_experiment(m1_toml; nogpu=nogpu, FT=FT)
    model2, x2 = unpack_experiment(m2_toml; nogpu=nogpu, FT=FT)
    model3, x3 = unpack_experiment(m3_toml; nogpu=nogpu, FT=FT)


    # Determine which `k` the images begin to differ from each other
    spectra1, k = batch_spectra(x1[:,:,:,1:nsamples] |> cpu)
    spectra2, k = batch_spectra(x2[:,:,:,1:nsamples] |> cpu)
    spectra3, k = batch_spectra(x3[:,:,:,1:nsamples] |> cpu)
    
    # Debiasing Bridge
    # The models use the same noising schedule, so the t_end is the same for each.
    N = FT(size(x1)[1])

    cutoff_idx = 3
    k_cutoff = FT(k[cutoff_idx])
    # This takes a mean over channels
    power_at_cutoff = FT(mean(spectra2[cutoff_idx,:,:]))
    t_end = FT(t_cutoff(power_at_cutoff, k_cutoff, N, model2.σ_max, model2.σ_min))
    
    init_x = x1[:,:,:,[3]]
    sol_m1_12, sol_m2_12 = diffusion_bridge_simulation(model1,
                                                       model2,
                                                       init_x,
                                                       nsteps;
                                                       forward_t_end=t_end,
                                                       forward_solver=DifferentialEquations.RK4(),
                                                       forward_sde=false,
                                                       reverse_solver=DifferentialEquations.RK4(),
                                                       reverse_t_end=t_end,
                                                       reverse_sde=false)
    
    # Superresolution Bridge
    # The models use the same noising schedule, so the t_end is the same for each.
    cutoff_idx = 8
    k_cutoff = FT(k[cutoff_idx])
    # This takes a mean over channels
    power_at_cutoff = FT(mean(spectra2[cutoff_idx,:,:]))
    t_end = FT(t_cutoff(power_at_cutoff, k_cutoff, N, model2.σ_max, model2.σ_min))

    sol_m2_23, sol_m3_23 = diffusion_bridge_simulation(model2,
                                                       model3,
                                                       sol_m2_12.u[end],
                                                       nsteps;
                                                       forward_t_end=t_end,
                                                       forward_solver=DifferentialEquations.RK4(),
                                                       forward_sde=false,
                                                       reverse_solver=DifferentialEquations.EM(),
                                                       reverse_t_end=t_end,
                                                       reverse_sde=true)

    # Rearrange the solution vectors into something we can plot
    # Also scale the images to be [0,1] for Grayscale plotting.
    # We do this per image, since the scale of the images grows during noising.
    # This is not ideal, probably.                                                 
    images12 = cat(sol_m1_12.u..., sol_m2_12.u[2:end]..., dims = (4));
    maxes = maximum(images12[:,:,[channel],:], dims = (1,2))
    mins = minimum(images12[:,:,[channel],:], dims = (1,2))
    images12 = @. (images12[:,:,[channel],:] - mins) / (maxes - mins)

    images23 = cat(sol_m2_23.u..., sol_m3_23.u[2:end]..., dims = (4));
    maxes = maximum(images23[:,:,[channel],:], dims = (1,2))
    mins = minimum(images23[:,:,[channel],:], dims = (1,2))
    images23 = @. (images23[:,:,[channel],:] - mins) / (maxes - mins)

    img_plot(images12, savepath, "debias.png"; ncolumns = size(images12)[end])
    img_plot(images23, savepath, "superresolve.png"; ncolumns = size(images23)[end])
 
    # Take a look at the spectra of all the "truth" images and the ones that we generated 
    s_debias, k = batch_spectra(sol_m2_12.u[end][:,:,[channel],:] |> cpu)
    s_sr, k = batch_spectra(sol_m3_23.u[end][:,:,[channel],:]|> cpu)
    plot(k, spectra3[:,channel,:][:], yaxis = :log, xaxis = :log, label = "truth")
    plot!(k, spectra2[:,channel,:][:], label = "blurry truth")
    plot!(k, spectra1[:,channel,:][:], label = "biased/blurry truth")
    plot!(k, s_sr[:], label = "generated truth")
    plot!(k, s_debias[:], label =" generated blurry truth")
    savefig(joinpath(savepath,"spectrum_comparison.png"))
end
