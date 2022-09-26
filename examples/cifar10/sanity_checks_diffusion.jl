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
include("../utils_plotting.jl")

if abspath(PROGRAM_FILE) == @__FILE__
    savedir = "./output"
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    ############################################################################
    # Issue loading function closures with BSON:
    # https://github.com/JuliaIO/BSON.jl/issues/69
    #
    BSON.@load checkpoint_path model model_smooth opt opt_smooth hparams
    #
    # BSON.@load does not work if defined inside plot_result(⋅) because
    # it contains a function closure, GaussFourierProject(⋅), containing W.
    ###########################################################################
    plot_result(model_smooth, savedir, hparams.data, num_steps=1000)
end
