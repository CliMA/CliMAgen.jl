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
include(joinpath(package_dir, "examples/utils_data.jl"))
include(joinpath(package_dir, "examples/utils_analysis.jl"))

function run_analysis(params; FT=Float32, logger=nothing)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    ntime = params.data.ntime
    resolution = params.data.resolution

    inchannels = params.model.inchannels

    nsamples = params.sampling.nsamples
    ngifs = params.sampling.ngifs
    nsteps = params.sampling.nsteps
    sampler = params.sampling.sampler

    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    # set up device
    if !nogpu && CUDA.has_cuda()
        dev = Flux.gpu
        @info "Sampling on GPU"
    else
        dev = Flux.cpu
        @info "Sampling on CPU"
    end

    # set up dataset
    dl, _ = get_data_mnist_3d(
        nsamples;
        FT=FT
    )
    xtrain, dl = Iterators.peel(dl)
    nspatial = 3

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = dev(model)

    # sample from the trained model
    t = ones(FT, nsamples) |> dev
    init_z = randn(FT, (resolution, resolution, ntime, inchannels, nsamples)) |> dev
    _, σ_T = CliMAgen.marginal_prob(model, zero(init_z), t)
    init_x = (σ_T .* init_z)
    time_steps = LinRange(1.0f0, 1.0f-3, nsteps)
    Δt = time_steps[1] - time_steps[2]
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt, nspatial=3)
    samples = cpu(samples)

    # loss curves
    loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    

    # make some gifs
    for i in 1:ngifs
        # xtrain    
        anim = @animate for j ∈ 1:size(xtrain, nspatial)
            heatmap(xtrain[:,:,j,1,i])
        end
        path = joinpath(savedir, "anim_fps15_train_$i.gif")
        gif(anim, path, fps = 15)

        # samples
        anim = @animate for j ∈ 1:size(samples, nspatial)
            heatmap(samples[:,:,j,1,i])
        end
        path = joinpath(savedir, "anim_fps15_gen_$(i).gif")
        gif(anim, path, fps = 15)
    end 
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    run_analysis(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
