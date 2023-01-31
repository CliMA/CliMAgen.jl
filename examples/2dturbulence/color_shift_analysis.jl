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

function sample(model, nsamples, nsteps, device, tilesize, inchannels)
    # sample from the trained model
    time_steps, Δt, init_x = setup_sampler(model,
                                          device,
                                          tilesize,
                                          inchannels;
                                          num_images=nsamples,
                                          num_steps=nsteps,
                                          )
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
    samples = cpu(samples)
    return samples
end

function color_shift_ablation(;
    toml="experiments/Experiment128_original.toml",
    figpath1 = "./scores_original.png",
    figpath2 = "./loss_original.png")
#    shift_input_toml="experiments/Experiment128_original_shift_input.toml"
#    mean_bypass_toml="experiments/Experiment128_original_mean_bypass.toml"
#    preprocess_toml="experiments/Experiment128_original_preprocess.toml"
#    all_mods_toml="experiments/Experiment128_original_mean_bypass_scale.toml"

    FT = Float32
    nogpu = false
    nsteps = 1000
    savepath = "./"
    nsamples = 100
    channel = 1
    model, xtrain = unpack_experiment(toml; FT = FT)
    samples = sample(model, nsamples, nsteps, gpu, size(xtrain)[1], size(xtrain)[3]) |> cpu
  
    for ch in [1,2]
        println("Generated μ ", Statistics.mean(samples, dims = (1,2,4))[:][ch], " ± ", Statistics.std(samples, dims = (1,2,4))[:][ch]/sqrt(nsamples))
        println("Data μ ", Statistics.mean(xtrain |>cpu, dims = (1,2,4))[:][ch], " ± ", Statistics.std(xtrain |> cpu, dims = (1,2,4))[:][ch]/sqrt(size(xtrain)[end]))
        println("Generated σ ", Statistics.std(samples, dims = (1,2,4))[:][ch])
        println("Data σ ", Statistics.std(xtrain |>cpu, dims = (1,2,4))[:][ch])
    end
    t, s_avg, s_dev = timewise_score(model, xtrain[:,:,:,1:nsamples], ϵ=1.0f-5)
    t, loss_avg, loss_dev = timewise_score_matching_loss(model, xtrain[:,:,:,1:nsamples], ϵ=1.0f-5)

    plot(t |> cpu, s_dev |> cpu, yaxis = :log, margin = 10Plots.mm, label = "Dev Score")
    plot!(t |> cpu, s_avg |> cpu,label ="Mean Score")
    plot!(t |> cpu, (1/model.σ_min) .* (model.σ_max ./model.σ_min) .^ (-1 .* t) |> cpu, label = "1/σ(t)")
    plot!(t |> cpu, (1/model.σ_min)/128 .* (model.σ_max ./model.σ_min) .^ (-1 .* t) |> cpu, label = "1/σ(t)/√N")
    savefig(figpath1)

    plot(t |> cpu, loss_dev |> cpu, yaxis = :log, ylim = [1e-6, 1e-1],margin = 10Plots.mm, label = "Dev Loss")
    plot!(t |> cpu, loss_avg |> cpu,label ="Mean Loss")
    savefig(figpath2)

   # ps = Flux.params(model)
   # lossfn = x -> score_matching_loss_variant_avg(model, x)
   # grad = Flux.gradient(() -> sum(lossfn(xtrain[:,:,:,1:64])), ps)
   ## @info norm(grad)
   # lossfn = x -> score_matching_loss_variant_dev(model, x)
   # grad = Flux.gradient(() -> sum(lossfn(xtrain[:,:,:,1:64])), ps)
   # @info norm(grad)
   x_0 = xtrain[:,:,:,64]
   t = LinRange(0.0f0,1.0f0,size(x_0)[end])
   if typeof(x_0) <:CuArray
       t = CuArray(t)
   end
   # sample from normal marginal
   z = randn!(similar(x_0))
   μ_t, σ_t = marginal_prob(model, x_0, t)
   x_t = @. μ_t + σ_t * z

   # evaluate model score s₀(𝘹(𝘵), 𝘵)
   s_t = CliMAgen.score(model, x_t, t)
   ϵ_t = s_t .* σ_t
   x_t_bar = Statistics.mean(x_t, dims = (1,2))
   dev = x_t .- x_t_bar
   s_t_without_mean = CliMAgen.score(model, dev, t)
   ϵ_t_no_mean = s_t_without_mean .* σ_t

   square(x) = x*x

   diff = ϵ_t_no_mean - ϵ_t
   diff_mean = Statistics.mean(diff, dims = (1,2))
   diff_dev = diff .- diff_mean

   diff_dev_avg = sqrt.((Statistics.mean(square.(diff_dev), dims = (1,2,3)) |> cpu)[:])
   diff_avg = sqrt.((Statistics.mean(square.(diff), dims = (1,2,3)) |> cpu)[:])
   ϵ̄ = Statistics.mean(ϵ_t, dims = (1,2))
   ϵ̄_no_mean = Statistics.mean(ϵ_t_no_mean, dims = (1,2))
   diff_avg_avg = sqrt.((Statistics.mean(square.(diff_mean),dims =(1,2,3)) |> cpu)[:])
   plot(t |> cpu, diff_avg,label = "√E[|ϵ(x) - ϵ(x')|^2]", xlabel = "Time", ylabel = "Magnitude")
  # plot!(t |> cpu, diff_dev_avg,label = "√E[|ϵ'(x) - ϵ'(x')|^2]", xlabel = "Time", ylabel = "Magnitude")
   #plot!(t |> cpu, diff_avg_avg, label = "√E[|ϵ̄(x)-ϵ̄(x')|^2]")
   plot!( margin = 20Plots.mm)
   savefig("./shifted_input.png")
   #plot(t |> cpu, Statistics.mean(abs.(ϵ_t), dims = (1,2,3))[:] |> cpu, label = "ϵ(x)")
   plot(t |> cpu, Statistics.mean(abs.(ϵ̄), dims = (3,))[:] |> cpu, label = "ϵ̄(x)")
   #plot!(t |> cpu, Statistics.mean(abs.(ϵ_t_no_mean), dims = (1,2,3))[:] |> cpu, label = "ϵ(x')")
   plot!(t |> cpu, Statistics.mean(abs.(ϵ̄_no_mean), dims = 3)[:] |> cpu, label = "ϵ̄(x')")

   plot!(margin = 10Plots.mm, legend = :bottomright)
   savefig("./shifted_input_eps.png")
end