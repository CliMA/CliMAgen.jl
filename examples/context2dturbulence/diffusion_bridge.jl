using BSON
using Flux
using CUDA
using cuDNN
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
    fraction::FT = FT(0.02)#params.data.fraction
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


function two_model_bridge(;
                          source_toml="experiments/Experiment_resize_64.toml",
                          target_toml="experiments/Experiment_preprocess_mixed.toml")
    FT = Float32
    device = Flux.gpu
    nsteps = 125
    savepath = "./output"
    tilesize = 512
    context_channels = 1
    noised_channels = 2
       
    wavenumber = FT(4)

    forward_model, xsource, csource, scaling_source = unpack_experiment(source_toml, wavenumber; device = device, FT=FT)
    reverse_model, xtarget, ctarget, scaling_target = unpack_experiment(target_toml, wavenumber; device = device,FT=FT)
                      
    # Determine which `k` the two sets of images begin to differ from each other
    source_spectra, k = batch_spectra((xsource |> cpu)[:,:,:,[end]])
    target_spectra, k = batch_spectra((xtarget |> cpu)[:,:,:,[end]])
                      
    # this is manual right now. just eyeball it.
    Plots.plot(Statistics.mean(source_spectra, dims = 2)[:], label = "resized 64x64")
    Plots.plot!(Statistics.mean(target_spectra, dims = 2)[:], label = "512x512")
    Plots.plot!(xlabel = "index", ylabel = "power", bottommargin = 10Plots.mm, leftmargin = 10Plots.mm, yaxis = :log, xaxis = :log, legend = :bottomleft)
    Plots.savefig("./tmp.png")
    cutoff_idx = 10
    k_cutoff = FT(k[cutoff_idx])
    N = FT(size(xsource)[1])
    source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))
    forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, N, forward_model.σ_max, forward_model.σ_min))
    target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, N, reverse_model.σ_max, reverse_model.σ_min))

    # create a bridge plot for a single image.
    idx = [5,10,15, 20, 25, 30]#size(xsource)[end]
    init_x = xsource[:,:,:,idx]
    reverse_context = ctarget[:,:,:,idx]
    forward_context = csource[:,:,:,idx]
    plotname = "sde_both_partial.png"
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
    # To make the plot with grayscale, we need to scale each one to between 0 and 1. 
    channel = 2
    images = cat(forward_solution.u..., reverse_solution.u[2:end]..., dims = (4));
    maxes = maximum(images, dims = (1,2,3))
    mins = minimum(images, dims = (1,2, 3))
    images = @. (images - mins) / (maxes - mins)
    #images = reshape(permutedims(images, [1,2,3,5,4]), (512, 512, 1, 21))
    img_plot(images[:,:,[2],:], savepath, plotname; ncolumns = 7)
    
end
