using HDF5, ProgressBars
hfile = h5open("data/data_1.0_0.0_0.0_0.0.hdf5")
tseries  = read(hfile["timeseries"])
close(hfile)


skip = 10
new_tseries = zeros(32, 32, 2, (2000 - skip) * 128)
rtseries = reshape(tseries, (32, 32, 128, 2000))
for i in ProgressBar(1:128)
    new_tseries[:, :, 2, 1 + (i-1)*(2000-skip):i*(2000-skip)] .= rtseries[:, :, i, 1:end-skip]
    new_tseries[:, :, 1, 1+(i-1)*(2000-skip):i*(2000-skip)] .= rtseries[:, :, i, 1+skip:end]
end

hfile = h5open("data/data_2.0_0.0_0.0_0.0.hdf5", "w")
hfile["timeseries"] = new_tseries
close(hfile)


skip = 30
new_tseries = zeros(32, 32, 2, (2000 - skip) * 128)
rtseries = reshape(tseries, (32, 32, 128, 2000))
for i in ProgressBar(1:128)
    new_tseries[:, :, 2, 1+(i-1)*(2000-skip):i*(2000-skip)] .= rtseries[:, :, i, 1:end-skip]
    new_tseries[:, :, 1, 1+(i-1)*(2000-skip):i*(2000-skip)] .= rtseries[:, :, i, 1+skip:end]
end

hfile = h5open("data/data_3.0_0.0_0.0_0.0.hdf5", "w")
hfile["timeseries"] = new_tseries
close(hfile)


skip = 50
new_tseries = zeros(32, 32, 2, (2000 - skip) * 128)
rtseries = reshape(tseries, (32, 32, 128, 2000))
for i in ProgressBar(1:128)
    new_tseries[:, :, 2, 1+(i-1)*(2000-skip):i*(2000-skip)] .= rtseries[:, :, i, 1:end-skip]
    new_tseries[:, :, 1, 1+(i-1)*(2000-skip):i*(2000-skip)] .= rtseries[:, :, i, 1+skip:end]
end

hfile = h5open("data/data_4.0_0.0_0.0_0.0.hdf5", "w")
hfile["timeseries"] = new_tseries
close(hfile)