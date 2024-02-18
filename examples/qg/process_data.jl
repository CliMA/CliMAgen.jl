using HDF5, ProgressBars, Statistics, Random
Random.seed!(1234)
#=
files = readdir()
qg_files = filter(x -> x[1] === 'q', files)
snapshots = zeros(128, 128, 4, length(qg_files))
for (i, qg_file) in ProgressBar(enumerate(qg_files))
    tmp_hfile = h5open(qg_file, "r")
    field1 = read(tmp_hfile["p1"]["value"])
    field2 = read(tmp_hfile["p2"]["value"])
    snapshots[:,:, 1, i] = (field1[1:128, 1:128] + field1[2:end, 1:128] + field1[1:128, 2:end] + field1[2:end, 2:end])/4
    snapshots[:,:, 2, i] = (field2[1:128, 1:128] + field2[2:end, 1:128] + field2[1:128, 2:end] + field2[2:end, 2:end])/4
    snapshots[:,:, 3, i] = read(tmp_hfile["q1t"]["value"])
    snapshots[:,:, 4, i] = read(tmp_hfile["q2t"]["value"])
    close(tmp_hfile)
end
hfile = h5open("data/data.hdf5", "w")
hfile["snapshots"] = snapshots
close(hfile)
=#

hfile = h5open("data/data.hdf5", "r")
snapshots = read(hfile["snapshots"])
close(hfile)
##
# Normalize
μs = mean(snapshots, dims = (1, 2, 4))
σs = 2 * std(snapshots, dims = (1, 2, 4))
normalized_snapshots = (snapshots .- μs) ./ σs



hfile0 = h5open("data_0.0_0.0_0.0_0.0.hdf5", "w")
hfile0["snapshots"] = Float32.(normalized_snapshots[:, :, 3:4, 1:end])
close(hfile0)

hfile1 = h5open("data_1.0_0.0_0.0_0.0_context.hdf5", "w")
hfile1["snapshots"] = Float32.(normalized_snapshots[:, :, [4, 3], 1:end])
close(hfile1)

hfile2 = h5open("data_2.0_0.0_0.0_0.0_context.hdf5", "w")
hfile2["snapshots"] =  Float32.(normalized_snapshots[:, :, [4, 1], 1:end])
close(hfile2)

##
inds1 = rand(1:size(snapshots)[end], 300)
inds2 = rand(1:size(snapshots)[end], 300)
m1 = maximum([norm(normalized_snapshots[:, :, 1, i] - normalized_snapshots[:, :, 1, j]) for i in inds1, j in inds2])
m2 = maximum([norm(normalized_snapshots[:, :, 2, i] - normalized_snapshots[:, :, 2, j]) for i in inds1, j in inds2])
m3 = maximum([norm(normalized_snapshots[:, :, 3, i] - normalized_snapshots[:, :, 3, j]) for i in inds1, j in inds2])
m4 = maximum([norm(normalized_snapshots[:, :, 4, i] - normalized_snapshots[:, :, 4, j]) for i in inds1, j in inds2])

file_sigma = h5open("sigmas.hdf5", "w")
file_sigma["sigmas"] = [m1, m2, m3, m4]
close(file_sigma)