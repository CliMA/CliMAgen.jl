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

function main(savedir)
    FT = Float32
    num_samples = 100
    num_images = 25

    # load checkpoint and params
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth hparams

    # load dataset used for training
    tile_size = hparams.data.size
    dl, _ = get_data_2dturbulence(
        hparams.data; 
        width=(tile_size, tile_size),
        stride=(tile_size, tile_size), 
        FT=FT
    )
    xtrain = cat([x for x in dl]..., dims=4)
    xtrain = xtrain[:, :, :, 1:num_samples]

    # sample from the trained model
    samples = generate_samples(model, hparams.data, num_samples, ; num_steps = 1000, ϵ=1.0f-5)
    
    # create plots with num_images images of sampled data and training data
    img_plot(samples.em[:,:,:,1:num_images], savedir, "em_images.png", hparams.data)
    img_plot(samples.pc[:,:,:,1:num_images], savedir, "pc_images.png", hparams.data)
    img_plot(xtrain[:,:,:,1:num_images], savedir, "train_images.png", hparams.data)

    # create plot showing distribution of spatial mean of generated and real images
    spatial_mean_plot(xtrain, samples.em, savedir, "spatial_mean_distribution.png")
    
    # create q-q plot for cumulants of pre-specified scalar statistics
    qq_plot(xtrain, samples.em, savedir, "qq_plot.png")

    # create plots for comparison of real vs. generated spectra
    spectrum_plot(xtrain, samples.em, savedir, "mean_spectra.png")

    # create a plot showing how the network as optimized over different SDE times
    t, loss = timewise_score_matching_loss(model, xtrain)
    plot(t, log.(loss), xlabel = "SDE time", ylabel ="log(loss)", label = "")
    savefig(joinpath(savedir, "loss.png"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main("output")
end
