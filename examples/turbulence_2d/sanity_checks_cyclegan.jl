using BSON
using Flux
using FFTW
using UnicodePlots

using Downscaling

# fix float type
const FT = Float32


data_dir = joinpath(pkgdir(Downscaling), "data")
output_dir = joinpath(pkgdir(Downscaling), "output/turbulence_2d/")
include(joinpath(data_dir, "utils_artifacts.jl"))

# obtain validation data and trained model
data = get_dataloader(Turbulence2D()).validation
model_path = joinpath(output_dir, "checkpoint_cyclegan.bson")
generator = BSON.load(model_path, @__MODULE__)[:model][1]

# make a plot
(a, b, noise), data = Base.Iterators.peel(data);
a, b, noise = (FT.(a), FT.(b), FT.(noise))
x = a[:, :, 1, 1];
y1 = b[:, :, 1, 1];
y2 = generator(a, noise)[:, :, 1, 1];
heatmap(x, width=64)
heatmap(y1, width=64)
heatmap(y2, width=64)
