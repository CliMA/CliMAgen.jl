using JLD2, HDF5
using Statistics
using LinearAlgebra

const regex = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$";

training_data_location = "/orcd/nese/raffaele/001/ssilvest/solution_ribased/"

files = readdir("training_data")

files = readdir(dir)
files = filter(x -> length(x) > 20, files)
files = filter(x -> x[1:20] == "compressed_iteration", files)
iterations = Int[]

for file in files
    file   = file[1:end-5]
    string = ""
    i = length(file)
    while occursin(regex, "$(file[i])")
        string = file[i] * string
        i -= 1
    end
    push!(iterations, parse(Int, string))
end

iterations = unique(iterations)
iterations = sort(iterations)
iterations = iterations[15:end]

number_of_files = length(iterations)

level1 = 80
level2 = 60
level3 = 40

patches = [101:228, 201:328]
snapshots = zeros(128, 128, 7, number_of_files)

for (i, iter) in enumerate(iterations)
    file = jldopen("data/compressed_iteration_$(iter).jld2")
    T = file["T"]
    S = file["S"]
    η = file["η"]

    snapshots[:, :, 1, i] = T[patches..., level1]
    snapshots[:, :, 2, i] = T[patches..., level2]
    snapshots[:, :, 3, i] = T[patches..., level3]
    snapshots[:, :, 4, i] = S[patches..., level1]
    snapshots[:, :, 5, i] = S[patches..., level2]
    snapshots[:, :, 6, i] = S[patches..., level3]
    snapshots[:, :, 7, i] = η[patches..., 1]
end

# Normalize
μs = mean(snapshots, dims = (1, 2, 4))
σs = 2 * std(snapshots, dims = (1, 2, 4))
normalized_snapshots = (snapshots .- μs) ./ σs

hfile3 = h5open("data/data_4.0_0.0_0.0_0.0_context.hdf5", "w")
hfile3["snapshots"] =  Float32.(normalized_snapshots[:, :, [2, 1], 1:end])
hfile3["mean"] = μs
hfile3["standard deviation"] = σs
close(hfile3)

inds1 = rand(1:size(snapshots)[end], min(number_of_files, 300))
inds2 = rand(1:size(snapshots)[end], min(number_of_files, 300))
m1 = maximum([norm(normalized_snapshots[:, :, 1, i] - normalized_snapshots[:, :, 1, j]) for i in inds1, j in inds2])
m2 = maximum([norm(normalized_snapshots[:, :, 2, i] - normalized_snapshots[:, :, 2, j]) for i in inds1, j in inds2])

file_sigma = h5open("sigmas_128.hdf5", "w")
file_sigma["sigmas"] = [m1, m2]
close(file_sigma)
