using UnicodePlots
using Statistics

using CliMAgen
include("../utils_data.jl")
include("../utils_plotting.jl")

function simple_stats_corner(imgs, path)
    # vector of means (over each image, for a given channel)
    # vector of medians
    # etc
    
    img_mean = Statistics.mean(imgs)
    img_median = Statistics.median(imgs)
    img_max = maximum(imgs)
    img_min = minimum(imgs)
    stats = (; min = min, max = max, img_max = img_max, img_mean=img_mean)
    corner(table)
end

function main()
    FT=Float32

    dl, _ = get_data_2dturbulence((batchsize=64,), FT=FT)
    xtrain = cat([x for x in dl]..., dims=4)

    max = maximum(xtrain, dims=(1, 2))
    min = minimum(xtrain, dims=(1, 2))
    μ = mean(xtrain, dims=(1, 2))
    σ = std(xtrain, dims=(1, 2))
    
    
#    Generate images
    savedir = "output"
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth hparams
    device = Flux.cpu
    @info "Using device: $device"
    model = model |> device
    time_steps, Δt, init_x = setup_sampler(model, device, hparams.data; num_images = 1, num_steps = 1000)
# Predictor Corrector
    pc = predictor_corrector_sampler(model, init_x, time_steps, Δt)
    img_max = maximum(pc, dims = (1,2))
    img_min = minimum(pc, dims = (1,2))
    img_μ = mean(pc, dims=(1, 2))
    img_σ = var(pc, dims=(1, 2))

end
