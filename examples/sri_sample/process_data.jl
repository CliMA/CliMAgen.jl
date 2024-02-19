using HDF5, Statistics, LinearAlgebra

hfile_true = h5open("data/data_0.0_0.0_0.0_0.0.hdf5", "r")
μ = read(hfile_true["mean"])
σ = read(hfile_true["standard deviation"])
close(hfile_true)


hfile_context_true = h5open("data/sri_data_context.hdf5", "r")
context_snapshots = read(hfile_context_true["snapshots"])
close(hfile_context_true)

μ =  mean(context_snapshots[:, :, 1, :])
std = std(context_snapshots[:, :, 1, :])

μ_context =  mean(context_snapshots[:, :, 2, :])
std_context = std(context_snapshots[:, :, 2, :])

r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = (context_snapshots[:, :, 1, :] .- μ) ./ σ
r_context_snapshots[:, :, 2, :] = (context_snapshots[:, :, 2, :] .- μ_context) ./ std_context

hfile_context = h5open("data/data_1.0_0.0_0.0_0.0_context.hdf5", "w")
hfile_context["snapshots"] = r_context_snapshots
hfile_context["mean"] = μ
hfile_context["standard deviation"] = σ
hfile_context["context mean"] = μ_context
hfile_context["context standard deviation"] = std_context
close(hfile_context)



skipit = 8
averaged_snapshots = mean([context_snapshots[i:skipit:end, j:skipit:end, 1, :] for i in 1:skipit, j in 1:skipit])[:, :, :]
numind = 64 ÷ skipit
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
μ_context =  mean(context_snapshots[:, :, 2, :])
std_context = std(context_snapshots[:, :, 2, :])

r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = (context_snapshots[:, :, 1, :] .- μ) ./ σ
r_context_snapshots[:, :, 2, :] = (context_snapshots[:, :, 2, :] .- μ_context) ./ std_context

hfile_context = h5open("data/data_2.0_0.0_0.0_0.0_context.hdf5", "w")
hfile_context["snapshots"] = r_context_snapshots
hfile_context["mean"] = μ
hfile_context["standard deviation"] = σ
hfile_context["context mean"] = μ_context
hfile_context["context standard deviation"] = std_context
close(hfile_context)



skipit = 16
averaged_snapshots = mean([context_snapshots[i:skipit:end, j:skipit:end, 1, :] for i in 1:skipit, j in 1:skipit])[:, :, :]
numind = 64 ÷ skipit
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
μ_context =  mean(context_snapshots[:, :, 2, :])
std_context = std(context_snapshots[:, :, 2, :])

r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = (context_snapshots[:, :, 1, :] .- μ) ./ σ
r_context_snapshots[:, :, 2, :] = (context_snapshots[:, :, 2, :] .- μ_context) ./ std_context

hfile_context = h5open("data/data_3.0_0.0_0.0_0.0_context.hdf5", "w")
hfile_context["snapshots"] = r_context_snapshots
hfile_context["mean"] = μ
hfile_context["standard deviation"] = σ
hfile_context["context mean"] = μ_context
hfile_context["context standard deviation"] = std_context
close(hfile_context)


skipit = 32
averaged_snapshots = mean([context_snapshots[i:skipit:end, j:skipit:end, 1, :] for i in 1:skipit, j in 1:skipit])[:, :, :]
numind = 64 ÷ skipit
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
μ_context =  mean(context_snapshots[:, :, 2, :])
std_context = std(context_snapshots[:, :, 2, :])

r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = (context_snapshots[:, :, 1, :] .- μ) ./ σ
r_context_snapshots[:, :, 2, :] = (context_snapshots[:, :, 2, :] .- μ_context) ./ std_context

hfile_context = h5open("data/data_4.0_0.0_0.0_0.0_context.hdf5", "w")
hfile_context["snapshots"] = r_context_snapshots
hfile_context["mean"] = μ
hfile_context["standard deviation"] = σ
hfile_context["context mean"] = μ_context
hfile_context["context standard deviation"] = std_context
close(hfile_context)


skipit = 64
averaged_snapshots = mean([context_snapshots[i:skipit:end, j:skipit:end, 1, :] for i in 1:skipit, j in 1:skipit])[:, :, :]
numind = 64 ÷ skipit
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
μ_context =  mean(context_snapshots[:, :, 2, :])
std_context = std(context_snapshots[:, :, 2, :])

r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = (context_snapshots[:, :, 1, :] .- μ) ./ σ
r_context_snapshots[:, :, 2, :] = (context_snapshots[:, :, 2, :] .- μ_context) ./ std_context

hfile_context = h5open("data/data_5.0_0.0_0.0_0.0_context.hdf5", "w")
hfile_context["snapshots"] = r_context_snapshots
hfile_context["mean"] = μ
hfile_context["standard deviation"] = σ
hfile_context["context mean"] = μ_context
hfile_context["context standard deviation"] = std_context
close(hfile_context)
