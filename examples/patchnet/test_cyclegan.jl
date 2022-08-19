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
generator_A = model[1]

(a, _) = first(data)

# Fake image generated in domain B
_, nx, ny, _, _ = size(a)
pi2 = div(nx, 2)+1:nx
pj2 = div(ny, 2)+1:ny
b11_fake = generator_A(a[1, :, :, :, :])[pi2, pj2, :, :]
b12_fake = generator_A(a[2, :, :, :, :])[pi2, pj2, :, :]
b21_fake = generator_A(a[3, :, :, :, :])[pi2, pj2, :, :]
b22_fake = generator_A(a[4, :, :, :, :])[pi2, pj2, :, :]
b_fake = cat(cat(b11_fake, b21_fake, dims=1), cat(b12_fake, b22_fake, dims=1), dims=2)

heatmap(a[4, :, :, 1, 1], height=32)
heatmap(b_fake[:, :, 1, 1], height=32)
