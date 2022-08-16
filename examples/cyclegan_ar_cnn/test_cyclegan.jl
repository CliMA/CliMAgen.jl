using BSON
using Flux
using UnicodePlots

using Downscaling


include("utils.jl")

field = "moisture"
path_to_data = "../../data/moist2d/moist2d_512x512.hdf5"
data = get_dataloader(path_to_data, field=field, split_ratio=0.5, batch_size=1).validation

model_path = joinpath(@__DIR__, "output/checkpoint_latest.bson")
model = BSON.load(model_path, @__MODULE__)[:model]
generator_A = model[1] |> gpu

(a, _) = first(data) |> gpu
heatmap(a[:, :, 1, 1] |> cpu, height=50)
heatmap(generator_A(a)[:, :, 1, 1] |> cpu, height=50)

