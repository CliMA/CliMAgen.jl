using BSON
using Flux
using UnicodePlots

using Downscaling


include("utils.jl")

function test(path)
    data = get_dataloader(path, split_ratio=0.5, batch_size=1).validation

    model_path = joinpath(@__DIR__, "output/checkpoint_latest.bson")
    model = BSON.load(model_path, @__MODULE__)[:model]
    generator_A = model[1]

    (a, _) = first(data)
    heatmap(a[:, :, 1, 1], height=32)
    heatmap(generator_A(a)[:, :, 1, 1], height=32)
end

# run if file is called directly but not if just included
if abspath(PROGRAM_FILE) == @__FILE__
    path_to_data = "../../data/moist2d/moist2d_512x512.hdf5"
    test(path_to_data)
end

