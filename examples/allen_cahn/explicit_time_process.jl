using HDF5
include("TimeseriesData.jl")
hfile = h5open("data/data_1.0_0.0_0.0_0.0.hdf5", "r")

timeseries = read(hfile["timeseries"]);
shape = read(hfile["timeseries shape"]);

close(hfile)

r_timeseries = reshape(timeseries, (32, 32, 1, 128, 2000))
τ0 = reshape(gfp(0.0), (32, 32, 1, 1))
τ1 = reshape(gfp(1.0), (32, 32, 1, 1))


inds0 = 1:2000
inds1 = (1+length(inds0)):(1970+length(inds0))
timeseries_context = zeros(32, 32, 3, length(inds0) + length(inds1))
timeseries_context[:, :, 1, inds0] .= r_timeseries[:, :, 1,  1, 1:end]
timeseries_context[:, :, 2, inds0] .= r_timeseries[:, :, 1,  1, 1:end]
timeseries_context[:, :, 3, inds0] .= τ0

timeseries_context[:, :, 1, inds1] .= r_timeseries[:, :, 1,  2, 1:end-30]
timeseries_context[:, :, 2, inds1] .= r_timeseries[:, :, 1,  2, 31:end]
timeseries_context[:, :, 3, inds1] .= τ1


hfile = h5open("data/data_5.0_0.0_0.0_0.0.hdf5", "w")
hfile["timeseries"] = timeseries_context
close(hfile)


##
inds0 = [1+2000*(j-1):2000*j for j in 1:64]
inds1 = [inds0[end][end]+1+1970*(j-1):inds0[end][end]+1970*j for j in 1:64]
timeseries_context = zeros(32, 32, 3, 2000*64 + 1970*64)
for (j,inds) in enumerate(inds0)
    timeseries_context[:, :, 1, inds] .= r_timeseries[:, :, 1,  j, 1:end]
    timeseries_context[:, :, 2, inds] .= r_timeseries[:, :, 1,  j, 1:end]
    timeseries_context[:, :, 3, inds] .= τ0
end

for (j, inds) in enumerate(inds1)
    timeseries_context[:, :, 1, inds] .= r_timeseries[:, :, 1,  j+64, 1:end-30]
    timeseries_context[:, :, 2, inds] .= r_timeseries[:, :, 1,  j+64, 31:end]
    timeseries_context[:, :, 3, inds] .= τ1
end

hfile = h5open("data/data_6.0_0.0_0.0_0.0.hdf5", "w")
hfile["timeseries"] = timeseries_context
close(hfile)
