using CairoMakie
using HDF5
include("utils.jl")
include("pixel_plots.jl")
savedir = "../output_standard_scaling_dropout/downscaling"
hdf5_path = joinpath(savedir,"samples_downscaled_smooth.hdf5")
fid = HDF5.h5open(hdf5_path, "r")
train_samples = HDF5.read(fid["downscaled_samples_train"]);
train_random_samples = HDF5.read(fid["random_samples_train"]);
train_data = HDF5.read(fid["data_train"]);
train_data_lores = HDF5.read(fid["data_train_lores"]);

test_samples = HDF5.read(fid["downscaled_samples_test"]);
test_random_samples = HDF5.read(fid["random_samples_test"]);
test_data = HDF5.read(fid["data_test"]);
test_data_lores = HDF5.read(fid["data_test_lores"]);
close(fid)

x_lr = train_data_lores
x_hr = train_data
s_hr = train_samples
r_hr = train_random_samples

pixel_plots(x_hr, s_hr, ["training", "downscaled"], joinpath(savedir,"downscaled_samples_train.png"))
pixel_plots(x_hr, r_hr, ["training", "random gen"], joinpath(savedir,"random_samples_train.png"))


clims = extrema(x_lr)
fig = Figure(resolution=(3250, 1200), fontsize=24)
ax = CairoMakie.Axis(fig[1,1], ylabel="Lo res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, x_lr[:,:,1,1], clims = clims)
ax = CairoMakie.Axis(fig[2,1], ylabel="Hi res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, x_hr[:,:,1,1], clims = clims)
ax = CairoMakie.Axis(fig[3,1], ylabel="Hi res fake", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, s_hr[:,:,1,1], clims = clims)
for i in 2:8
    ax = CairoMakie.Axis(fig[1,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, x_lr[:,:,1,i*8], clims = clims)
    ax = CairoMakie.Axis(fig[2,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, x_hr[:,:,1,i*8], clims = clims)
    ax = CairoMakie.Axis(fig[3,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, s_hr[:,:,1,i*8], clims = clims)
end
save(joinpath(savedir,"downscaling_train.png"), fig, px_per_unit = 2)



x_lr = test_data_lores
x_hr = test_data
s_hr = test_samples
r_hr = test_random_samples

pixel_plots(x_hr, s_hr, ["test data", "downscaled"], joinpath(savedir,"downscaled_samples_test.png"))
pixel_plots(x_hr, r_hr, ["test data", "random gen"], joinpath(savedir,"random_samples_test.png"))


clims = extrema(x_lr)
fig = Figure(resolution=(3250, 1200), fontsize=24)
ax = CairoMakie.Axis(fig[1,1], ylabel="Lo res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, x_lr[:,:,1,1], clims = clims)
ax = CairoMakie.Axis(fig[2,1], ylabel="Hi res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, x_hr[:,:,1,1], clims = clims)
ax = CairoMakie.Axis(fig[3,1], ylabel="Hi res fake", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, s_hr[:,:,1,1], clims = clims)
for i in 2:8
    ax = CairoMakie.Axis(fig[1,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, x_lr[:,:,1,i*8], clims = clims)
    ax = CairoMakie.Axis(fig[2,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, x_hr[:,:,1,i*8], clims = clims)
    ax = CairoMakie.Axis(fig[3,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, s_hr[:,:,1,i*8], clims = clims)
end
save(joinpath(savedir,"downscaling_test.png"), fig, px_per_unit = 2)


