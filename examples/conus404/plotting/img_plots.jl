using CairoMakie
using HDF5
hdf5_path = "output_standard_scaling_dropout/samples_downscaled_smooth.hdf5"
fid = HDF5.h5open(hdf5_path, "r")
train_samples = HDF5.read(fid["generated_samples_train"])
train_data = HDF5.read(fid["data_train"])
train_data_lores = HDF5.read(fid["data_train_lores"])

test_samples = HDF5.read(fid["generated_samples_test"])
test_data = HDF5.read(fid["data_test"])
test_data_lores = HDF5.read(fid["data_test_lores"])
close(fid)

x_lr = train_data_lores
x_hr = train_data
s_hr = train_samples


clims = extrema(x_lr)
fig = Figure(resolution=(1200, 1800), fontsize=24)
ax = CairoMakie.Axis(fig[1,1], title="Lo res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, x_lr[:,:,1,1], clims = clims)
ax = CairoMakie.Axis(fig[1,2], title="Hi res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, x_hr[:,:,1,1], clims = clims)
ax = CairoMakie.Axis(fig[1,3], title="Hi res fake", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, s_hr[:,:,1,1], clims = clims)
for i in 2:4
    ax = CairoMakie.Axis(fig[i,1], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, x_lr[:,:,1,i*16], clims = clims)
    ax = CairoMakie.Axis(fig[i,2], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, x_hr[:,:,1,i*16], clims = clims)
    ax = CairoMakie.Axis(fig[i,3], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, s_hr[:,:,1,i*16], clims = clims)
end
save("output_standard_scaling_dropout/downscaling_train.png", fig, px_per_unit = 2)



x_lr = test_data_lores
x_hr = test_data
s_hr = test_samples


clims = extrema(x_lr)
fig = Figure(resolution=(1200, 1800), fontsize=24)
ax = CairoMakie.Axis(fig[1,1], title="Lo res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, x_lr[:,:,1,1], clims = clims)
ax = CairoMakie.Axis(fig[1,2], title="Hi res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, x_hr[:,:,1,1], clims = clims)
ax = CairoMakie.Axis(fig[1,3], title="Hi res fake", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
heatmap!(ax, s_hr[:,:,1,1], clims = clims)
for i in 2:4
    ax = CairoMakie.Axis(fig[i,1], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, x_lr[:,:,1,i*16], clims = clims)
    ax = CairoMakie.Axis(fig[i,2], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, x_hr[:,:,1,i*16], clims = clims)
    ax = CairoMakie.Axis(fig[i,3], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    heatmap!(ax, s_hr[:,:,1,i*16], clims = clims)
end
save("output_standard_scaling_dropout/downscaling_test.png", fig, px_per_unit = 2)

