using CairoMakie
using DataFrames
using Statistics
using CSV

base_path = "/groups/esm/kdeck/fashion_mnist_output_paper/fashion_mnist"
epochs = 100
smooth = true
sm = smooth ? "/smooth/" : "/"

# fig
ch = 1
nsamples = 200
nsamples_256 = 100
nsamples_512 = 33
fig = Figure(resolution=(1600, 400), fontsize=24)
min_x = 0.5
max_x = 5.5
min_y = -0.5
max_y = 2.0
xticks = (0.5:0.5:5.5, ["", "32", "", "64", "", "128", "", "256", "", "512", ""])


# Means
cumu = 1
# baseline
x = Int[]
y = Float32[]
dodge = Int[]
side = Symbol[]

res = "32x32"
i = 1
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "64x64"
i = 2
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "128x128"
i = 3
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "256x256"
i = 4
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples_256), 2ones(Int, nsamples_256))
side = vcat(side, [:left for _ in 1:nsamples_256], [:right for _ in 1:nsamples_256])
x = vcat(x, i*ones(Int, 2nsamples_256))

res = "512x512"
i = 5
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples_512), 2ones(Int, nsamples_512))
side = vcat(side, [:left for _ in 1:nsamples_512], [:right for _ in 1:nsamples_512])
x = vcat(x, i*ones(Int, 2nsamples_512))

color = @. ifelse(side === :left, :orange, :teal)

ax = Axis(fig[1,1], xlabel="Resolution", ylabel="Spatial mean", title="Baseline Net, Mean", xticks=xticks, titlealign=:left)
violin!(ax, x, y, side = side, color = color, label="data")
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

# modification on
x = Int[]
y = Float32[]
dodge = Int[]
side = Symbol[]

res = "32x32"
i = 1
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "64x64"
i = 2
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "128x128"
i = 3
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "256x256"
i = 4
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples_256), 2ones(Int, nsamples_256))
side = vcat(side, [:left for _ in 1:nsamples_256], [:right for _ in 1:nsamples_256])
x = vcat(x, i*ones(Int, 2nsamples_256))

res = "512x512"
i = 5
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples_512), 2ones(Int, nsamples_512))
side = vcat(side, [:left for _ in 1:nsamples_512], [:right for _ in 1:nsamples_512])
x = vcat(x, i*ones(Int, 2nsamples_512))

color = @. ifelse(side === :left, :orange, :teal)

ax = Axis(fig[1,2], xlabel="Resolution", title="Modified Net, Mean", xticks=xticks, yticklabelsvisible=false, titlealign=:left)
p = violin!(ax, x, y, side = side, color = color, label=["data", "generated"])
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)


# Stds
cumu = 2
min_y = -0.2
max_y = 0.8
# baseline
x = Int[]
y = Float32[]
dodge = Int[]
side = Symbol[]


res = "32x32"
i = 1
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "64x64"
i = 2
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "128x128"
i = 3
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "256x256"
i = 4
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples_256), 2ones(Int, nsamples_256))
side = vcat(side, [:left for _ in 1:nsamples_256], [:right for _ in 1:nsamples_256])
x = vcat(x, i*ones(Int, 2nsamples_256))

res = "512x512"
i = 5
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples_512), 2ones(Int, nsamples_512))
side = vcat(side, [:left for _ in 1:nsamples_512], [:right for _ in 1:nsamples_512])
x = vcat(x, i*ones(Int, 2nsamples_512))

color = @. ifelse(side === :left, :orange, :teal)

ax = Axis(fig[1,3], xlabel="Resolution", ylabel="Spatial standard deviation", title="Baseline Net, Std. Dev.", xticks=xticks, titlealign=:left)
violin!(ax, x, y, side = side, color = color, label="data")
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

# modification on
x = Int[]
y = Float32[]
dodge = Int[]
side = Symbol[]

res = "32x32"
i = 1
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "64x64"
i = 2
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "128x128"
i = 3
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples), 2ones(Int, nsamples))
side = vcat(side, [:left for _ in 1:nsamples], [:right for _ in 1:nsamples])
x = vcat(x, i*ones(Int, 2nsamples))

res = "256x256"
i = 4
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples_256), 2ones(Int, nsamples_256))
side = vcat(side, [:left for _ in 1:nsamples_256], [:right for _ in 1:nsamples_256])
x = vcat(x, i*ones(Int, 2nsamples_256))

res = "512x512"
i = 5
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_off$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== true) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
filepath = joinpath(base_path, "Experiment_" * res * "_all_mods_on$(sm)cumulants.csv")
df = DataFrame(CSV.File(filepath))
y = vcat(y, df[(df.isreal .== false) .&& (df.channel .== ch) .&& (df.cumulant .== cumu), :].values)
dodge = vcat(dodge, ones(Int, nsamples_512), 2ones(Int, nsamples_512))
side = vcat(side, [:left for _ in 1:nsamples_512], [:right for _ in 1:nsamples_512])
x = vcat(x, i*ones(Int, 2nsamples_512))

color = @. ifelse(side === :left, :orange, :teal)

ax = Axis(fig[1,4], xlabel="Resolution", title="Modified Net, Std. Dev.", xticks=xticks, yticklabelsvisible=false, titlealign=:left)
p = violin!(ax, x, y, side = side, color = color, label=["data", "generated"])
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

# legend
ccolors = [:orange, :teal]
elems = [[MarkerElement(color = col, marker=:circle, markersize = 15, strokecolor = :black)] for col in ccolors]
axislegend(ax, elems, ["Real data", "Generated data"]; position=:rt, labelsize=16)


if smooth
    save("fig_mean_std_compact_epochs$(epochs)_smooth.png", fig, px_per_unit = 2)
else
    save("fig_mean_std_compact_epochs$epochs.png", fig, px_per_unit = 2)
end