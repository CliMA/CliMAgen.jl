using CairoMakie
using HDF5
include("utils.jl")
include("pixel_plots.jl")
# Specific to this experiment
output_dir = "output_tmax_zmuv_lowpass"
clims_tuple = ((250,320),(-9,-2))
inchannels = [1]
precip_channel = 2
names = ["Temperature", "Precipitation"]

savedir = "../$(output_dir)/downscaling"
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


include("img_plots.jl")
include("spectra.jl")