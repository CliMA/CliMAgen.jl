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

if abspath(PROGRAM_FILE) == @__FILE__
    FT = Float32
    savedir = "examples/mnist/output"
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    ############################################################################
    # Issue loading function closures with BSON:
    # https://github.com/JuliaIO/BSON.jl/issues/69
    #
    BSON.@load checkpoint_path model model_smooth opt opt_smooth hparams

    # BSON.@load does not work if defined inside plot_result(⋅) because
    # it contains a function closure, GaussFourierProject(⋅), containing W.
    ###########################################################################
    
    device = Flux.cpu
    num_samples = 100
    num_images = 25

    dl, _ = get_data_mnist(hparams.data; FT=FT)
    xtrain = cat([x for x in dl]..., dims=4)
    xtrain = xtrain[:, :, :, 1:num_samples]

    samples = generate_samples(model_smooth, hparams.data, device, num_samples, ; num_steps = 500)
    img_plot(samples.em[:,:,:,1:num_images], savedir, "em_images.png", hparams.data)
    img_plot(samples.pc[:,:,:,1:num_images], savedir, "pc_images.png", hparams.data)
    img_plot(xtrain[:,:,:,1:num_images], savedir, "train_images.png", hparams.data)

    qq_plot(xtrain, samples.pc, savedir, "qq_plot.png")
end
