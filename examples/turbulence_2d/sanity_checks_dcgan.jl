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
model_path = joinpath(output_dir, "checkpoint_dcgan.bson")
generator = BSON.load(model_path, @__MODULE__)[:model][1]

# make a plot
(_, b, _), data = Base.Iterators.peel(data);
a, b = (rand(FT, size(b)...), FT.(b));
y1 = b[:, :, 1, 1]
y2 = generator(a)[:, :, 1, 1]
heatmap(y1, width=64)
heatmap(y2, width=64)
