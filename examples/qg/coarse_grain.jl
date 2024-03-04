using HDF5, LinearAlgebra, Statistics

hfile = h5open("data/data_3.0_0.0_0.0_0.0_context.hdf5")
context_snapshots = read(hfile["snapshots"])
close(hfile)
μ = 0.0 
σ = 0.5
##
skipit = 4
averaged_snapshots = mean([context_snapshots[i:skipit:end, j:skipit:end, 2, :] for i in 1:skipit, j in 1:skipit])[:, :, :]
numind = 128 ÷ skipit
new_context = copy(context_snapshots)
for i in 1:numind, j in 1:numind 
    start_i = 1 + (i-1)*skipit
    end_i = i * skipit
    start_j = 1 + (j-1)*skipit
    end_j = j * skipit
    for k in 1:size(new_context)[end]
        new_context[start_i:end_i, start_j:end_j, 2, k] .= averaged_snapshots[i,j, k]
    end
end
context_snapshots[:, :, 2, :] .= new_context[:, :, 2, :]
r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = context_snapshots[:, :, 1, :] 
r_context_snapshots[:, :, 2, :] = context_snapshots[:, :, 2, :]

hfile_context = h5open("data/data_5.0_0.0_0.0_0.0_context.hdf5", "w")
hfile_context["snapshots"] = r_context_snapshots
hfile_context["mean"] = μ
hfile_context["standard deviation"] = σ
close(hfile_context)



skipit = 16
averaged_snapshots = mean([context_snapshots[i:skipit:end, j:skipit:end, 2, :] for i in 1:skipit, j in 1:skipit])[:, :, :]
numind = 128 ÷ skipit
new_context = copy(context_snapshots)
for i in 1:numind, j in 1:numind 
    start_i = 1 + (i-1)*skipit
    end_i = i * skipit
    start_j = 1 + (j-1)*skipit
    end_j = j * skipit
    for k in 1:size(new_context)[end]
        new_context[start_i:end_i, start_j:end_j, 2, k] .= averaged_snapshots[i,j, k]
    end
end
context_snapshots[:, :, 2, :] .= new_context[:, :, 2, :]
r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = context_snapshots[:, :, 1, :] 
r_context_snapshots[:, :, 2, :] = context_snapshots[:, :, 2, :]

hfile_context = h5open("data/data_6.0_0.0_0.0_0.0_context.hdf5", "w")
hfile_context["snapshots"] = r_context_snapshots
hfile_context["mean"] = μ
hfile_context["standard deviation"] = σ
close(hfile_context)



skipit = 64
averaged_snapshots = mean([context_snapshots[i:skipit:end, j:skipit:end, 2, :] for i in 1:skipit, j in 1:skipit])[:, :, :]
numind = 128 ÷ skipit
new_context = copy(context_snapshots)
for i in 1:numind, j in 1:numind 
    start_i = 1 + (i-1)*skipit
    end_i = i * skipit
    start_j = 1 + (j-1)*skipit
    end_j = j * skipit
    for k in 1:size(new_context)[end]
        new_context[start_i:end_i, start_j:end_j, 2, k] .= averaged_snapshots[i,j, k]
    end
end
context_snapshots[:, :, 2, :] .= new_context[:, :, 2, :]
r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = context_snapshots[:, :, 1, :] 
r_context_snapshots[:, :, 2, :] = context_snapshots[:, :, 2, :]

hfile_context = h5open("data/data_7.0_0.0_0.0_0.0_context.hdf5", "w")
hfile_context["snapshots"] = r_context_snapshots
hfile_context["mean"] = μ
hfile_context["standard deviation"] = σ
close(hfile_context)
