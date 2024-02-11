using HDF5, Statistics

hfile_true = h5open("data/data_0.0_0.0_0.0_0.0.hdf5", "r")
μ = read(hfile_true["mean"])
σ = read(hfile_true["standard deviation"])
close(hfile_true)


hfile_context_true = h5open("data/sri_data_context.hdf5", "r")
context_snapshots = read(hfile_context_true["snapshots"])
close(hfile_context_true)

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