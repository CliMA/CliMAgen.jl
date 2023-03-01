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

function unpack_experiment(experiment_toml, label; nogpu = false, FT=Float32)
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    rngseed = params.experiment.rngseed
    savedir = params.experiment.savedir
    
    noised_channels = params.model.noised_channels
    context_channels = params.model.context_channels

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

    dl, _ = get_data_context2dturbulence(
        batchsize;
        width=(tilesize, tilesize),
        stride=(tilesize, tilesize),
        label=label,
        FT=FT
    )
    train = cat([x for x in dl]..., dims=4)

    xtrain = train[:,:,1:noised_channels,:]
    ctrain = train[:,:,(noised_channels+1):(noised_channels+context_channels),:]

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    return model, xtrain |> device, ctrain |> device
end


function two_model_bridge(label;
                          source_toml="experiments/Experiment256_larger_bias_and_blurry.toml",
                          target_toml="experiments/Experiment256_base_copy.toml")
    FT = Float32
    nogpu = false
    nsteps = 125
    savepath = "./"
    nsamples = 10
    channel = 1

    forward_model, xsource, csource = unpack_experiment(source_toml, label; nogpu=nogpu, FT=FT)
    reverse_model, xtarget, ctarget = unpack_experiment(target_toml, label; nogpu=nogpu, FT=FT)

    # Determine which `k` the two sets of images begin to differ from each other
    
    source_spectra, k = batch_spectra(xsource |> cpu, size(xsource)[1])
    target_spectra, k = batch_spectra(xtarget |> cpu, size(xtarget)[1])

    # this is manual right now. just eyeball it.
    cutoff_idx = 3
    k_cutoff = FT(k[cutoff_idx])

    source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))
    forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, forward_model.σ_max, forward_model.σ_min))
    target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, reverse_model.σ_max, reverse_model.σ_min))

    # create a bridge plot for a single image.
    idx = 3
    init_x = xsource[:,:,:,[idx]]
    reverse_context = ctarget[:,:,:,[idx]]
    forward_context = csource[:,:,:,[idx]]
    plotname = "sde_reverse_partial.png"
    forward_solution, reverse_solution = diffusion_bridge_simulation(forward_model,
                                                                     reverse_model,
                                                                     init_x,
                                                                     nsteps;
                                                                     forward_c=forward_context,
                                                                     reverse_c=reverse_context,
                                                                     FT = FT,
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
