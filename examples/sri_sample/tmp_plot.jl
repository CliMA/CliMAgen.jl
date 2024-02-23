using HDF5, Statistics, LinearAlgebra, GLMakie

hfile_context_true = h5open("data/sri_data_context.hdf5", "r")
context_snapshots = read(hfile_context_true["snapshots"])
close(hfile_context_true)

μ =  mean(context_snapshots[:, :, 1, :])
σ = std(context_snapshots[:, :, 1, :])

μ_context =  mean(context_snapshots[:, :, 2, :])
std_context = std(context_snapshots[:, :, 2, :])

r_context_snapshots = copy(context_snapshots)
r_context_snapshots[:, :, 1, :] = (context_snapshots[:, :, 1, :] .- μ) ./ σ
r_context_snapshots[:, :, 2, :] = (context_snapshots[:, :, 2, :] .- μ_context) ./ std_context



hfile_context_true = h5open("data/sri_data_context_old.hdf5", "r")
context_snapshots = read(hfile_context_true["snapshots"])
close(hfile_context_true)

μ =  mean(context_snapshots[:, :, 1, :])
σ = std(context_snapshots[:, :, 1, :])

μ_context =  mean(context_snapshots[:, :, 2, :])
std_context = std(context_snapshots[:, :, 2, :])

r_context_snapshots_2 = copy(context_snapshots)
r_context_snapshots_2[:, :, 1, :] = (context_snapshots[:, :, 1, :] .- μ) ./ σ
r_context_snapshots_2[:, :, 2, :] = (context_snapshots[:, :, 2, :] .- μ_context) ./ std_context


fig = Figure(resolution = (900, 900))
cr = extrema(r_context_snapshots[:,:, 1, 1][:])
ax = GLMakie.Axis(fig[1, 1])
GLMakie.heatmap!(ax, r_context_snapshots[:,:, 2, 1], colormap = :thermal, colorrange = cr)
hidedecorations!(ax)
ax = GLMakie.Axis(fig[1, 2])
GLMakie.heatmap!(ax, r_context_snapshots_2[:,:, 2, 1], colormap = :thermal, colorrange = cr)
display(fig)


