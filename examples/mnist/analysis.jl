using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics

using CliMAgen
include("../utils_data.jl")
include("../utils_analysis.jl")

function main()
    FT = Float32
    num_samples = 100
    num_images = 25

    # directory for saving things
    savedir = "output"

    # load checkpoint and params
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth hparams
 # hyperparameters
    hparams = HyperParameters(
        data = (;
                nbatch  = 64,
                inchannels = 1,
                size = 32,
                ),
        model = (; 
                 σ_max   = FT(4.66),
                 σ_min   = FT(0.466),
                 ),
        optimizer = (;     
            lr      = FT(0.0002),
            ϵ       = FT(1e-8),
            β1      = FT(0.9),
            β2      = FT(0.999),
            nwarmup = 1,
            ema_rate = FT(0.999),
        ),
        training = (; 
            nepochs = 30,
        )
    )
    # load dataset used for training
    dl, _ = get_data_mnist(
        hparams.data; 
        FT=FT
    )
    xtrain = cat([x for x in dl]..., dims=4)
    xtrain = xtrain[:, :, :, 1:num_samples]

    # sample from the trained model
    samples = generate_samples(model, hparams.data, num_samples, ; num_steps = 500, ϵ=0.1f0)
    
    # create plots with num_images images of sampled data and training data
    img_plot(samples.em[:,:,:,1:num_images], savedir, "em_images.png", hparams.data)
    img_plot(samples.pc[:,:,:,1:num_images], savedir, "pc_images.png", hparams.data)
    img_plot(xtrain[:,:,:,1:num_images], savedir, "train_images.png", hparams.data)

    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain, samples.em, savedir, "qq_plot.png")

    # create plots for comparison of real vs. generated spectra
    spectrum_plot(xtrain, samples.em, savedir, "mean_spectra.png")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

function timewise_score_matching_loss(model, x_0, ϵ=1.0f-5)
    # sample times
    t = LinRange(0.0f0,1.0f0,size(xtrain)[end])

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = CliMAgen.marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = CliMAgen.score(model, x_t, t)

    # Assume that λ(t) = σ(t)² and pull it into L₂-norm
    # Below, z / σ_t = -∇ log [𝒫₀ₜ(𝘹(𝘵) | 𝘹(0))
    loss = @. (z + σ_t * s_t)^2 # squared deviations from real score
    loss = sum(loss, dims=1:(ndims(x_0)-1)) # L₂-norm

    return t, loss[:]
end

function model_scale(model, x_0, ϵ=1.0f-5)
    # sample times
    t = LinRange(0.0f0, 1.0f0, size(xtrain)[end])

    # sample from normal marginal
    z = randn!(similar(x_0))
    μ_t, σ_t = CliMAgen.marginal_prob(model, x_0, t)
    x_t = @. μ_t + σ_t * z

    # evaluate model score s₀(𝘹(𝘵), 𝘵)
    s_t = CliMAgen.score(model, x_t, t)
    scale = sqrt.(sum((σ_t .* s_t).^2, dims=1:(ndims(x_0)-1))) # L₂-norm

    return t, scale[:]
end
