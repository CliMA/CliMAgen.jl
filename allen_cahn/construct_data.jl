using HDF5, ProgressBars, Random
Random.seed!(1234)
include("TimeseriesData.jl")

hfile = h5open("data/data_1.0_0.0_0.0_0.0.hdf5")
tseries  = read(hfile["timeseries"])
shape = read(hfile["timeseries shape"])
close(hfile)

rtseries = reshape(tseries, Tuple(shape))

gfp = GaussianFourierProjection(32, 32, 30.0f0)

timelags = [0, 30, 1000]
ensemble_members = 4
unfolded_shapes = shape[end] .- timelags

index_jumps = [0, cumsum(unfolded_shapes)...]

new_series = zeros(shape[1], shape[2], 3, sum(unfolded_shapes), ensemble_members);

for (j, τ) in ProgressBar(enumerate(timelags))
    t = gfp(τ/timelags[end])
    for i in 1:ensemble_members 
        new_series[:, :, 1, 1+index_jumps[j]:index_jumps[j+1], i] .= rtseries[:, :, 1, i, 1:end-τ];
        new_series[:, :, 2, 1+index_jumps[j]:index_jumps[j+1], i] .= rtseries[:, :, 1, i, 1+τ:end];
        new_series[:, :, 3, 1+index_jumps[j]:index_jumps[j+1], i] .= reshape(t, (32, 32, 1));
    end
end

r_new_series = reshape(new_series, (shape[1], shape[2], 3, sum(unfolded_shapes) * ensemble_members));


hfile = h5open("data/data_8.0_0.0_0.0_0.0.hdf5", "w")
hfile["timeseries"] = r_new_series
close(hfile)


