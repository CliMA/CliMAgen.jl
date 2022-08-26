using BSON
using Flux
using FFTW
using UnicodePlots

using Downscaling

# fix float type
const FT = Float32


data_dir = joinpath(pkgdir(Downscaling), "data")
output_dir = joinpath(pkgdir(Downscaling), "output/kuramoto_sivashinsky/")
include(joinpath(data_dir, "utils_artifacts.jl"))

# obtain validation data and trained model
data = get_dataloader(KuramotoSivashinsky()).validation
model_path = joinpath(output_dir, "checkpoint_cyclegan.bson")
generator = BSON.load(model_path, @__MODULE__)[:model][1]

# make a plot
(a, b, noise), data = Base.Iterators.peel(data);
a, b, noise = (FT.(a), FT.(b), FT.(noise))
x = 1:size(b)[1]
y1 = b[:, 1, 1]
y2 = generator(a, noise)[:, 1, 1]
plt = lineplot(x, y1, width=64, height=32);
lineplot!(plt, x, y2)

# spectral computation
nx = length(y1)
inds = vcat(div(nx, 2)+2:nx, 1:div(nx, 2)+1)
k = vcat(0:div(nx, 2), -div(nx, 2)+1:-1)
y1_fft = log10.(abs.(fft(y1)) .+ eps(FT))
y2_fft = log10.(abs.(fft(y2)) .+ eps(FT))
plt = lineplot(k[inds], y1_fft[inds], width=64, height=32)
lineplot!(plt, k[inds], y2_fft[inds])
