using BSON
using Flux
using FFTW
using UnicodePlots
using FFTW
using Base.Iterators
using DataFrames

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


mpty_matrix = Matrix{Float64}(undef, 64,64)
empty_matrix  .= 0.0
mean_fft = copy(empty_matrix )
mean_fft_squared =  copy(empty_matrix)
for i in 1:10
    (elem, data) = Base.Iterators.peel(data)

    (_, b, _) = elem
    spectrum = abs.(fft(b .- mean(b))[1:64,1:64,1,1])
    mean_fft .+= spectrum
    mean_fft_squared .+= spectrum.^2.0
end
mean_fft = mean_fft ./ 10.0
std_fft = sqrt.(mean_fft_squared ./10.0 - mean_fft.^2.0)
std_error_fft = std_fft ./sqrt(10.0)

amplitude = log10.(mean_fft[1:64, 1:64,1,1] .+ eps(1.0))[2:end]
values = copy(empty_matrix)
for i in 1:64
    for j in 1:64
        values[i,j] = log10(sqrt((i-1)^2.0+(j-1.0)^2.0) +eps(1.0))
    end
end
values = values[2:end]
radii = Array(0.0:0.05:1.95)
bins = [argmin(abs.(radius .- radii)) for radius in values]
df = DataFrame(amplitude = amplitude, radius_bin = bins, radius = values)
gdf = groupby(df, :radius_bin)
counts = combine(gdf, nrow)
means = combine(gdf, [:amplitude, :radius] => ((p, s) -> (amplitude=mean(p), radius=mean(s))) => AsTable)
stds = combine(gdf, [:amplitude, :radius] => ((p, s) -> (amplitude=std(p), radius=std(s))) => AsTable)
std_errors = stds.amplitude ./ sqrt.(counts.nrow)
std_errors[isnan.(std_errors)] .= means.amplitude[isnan.(std_errors)]
plot(means.radius, means.amplitude, ribbon = std_errors, xlabel = "|k|", ylabel = "P(k)", label = "")
