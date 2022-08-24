using BSON
using Flux
using UnicodePlots

using Downscaling
examples_dir = joinpath(pkgdir(Downscaling), "examples")
cyclegan_dir = joinpath(examples_dir, "cyclegan_noise")
include(joinpath(cyclegan_dir, "utils.jl"))
include(joinpath(examples_dir, "artifact_utils.jl"))

field = "moisture"
local_dataset_directory = obtain_local_dataset_path(examples_dir, moist2d.dataname, moist2d.url, moist2d.filename)
local_dataset_path = joinpath(local_dataset_directory, moist2d.filename)
data = get_dataloader(local_dataset_path, field=field, split_ratio=0.5, batch_size=1).validation

output_dir = joinpath(cyclegan_dir, "output")
output_filepath = joinpath(output_dir, "checkpoint_latest.bson")

model = BSON.load(output_filepath, @__MODULE__)[:model]
generator_A = model[1]

a, _, noise = first(data)
noise = rand(eltype(a), size(noise))


heatmap(a[:, :, 1, 1], height=32)
heatmap(generator_A(a, noise)[:, :, 1, 1], height=32)
