using GLMakie, HDF5, Random, Statistics
Random.seed!(1234)


hfile_sample = h5open("data/data_2.0_0.0_0.0_0.0_context_analysis.hdf5", "r")
hfile_context = h5open("data/data_2.0_0.0_0.0_0.0_context.hdf5", "r")

snapshots = read(hfile_context["snapshots"])
#=
μ = read(hfile_context["mean"])
σ = read(hfile_context["standard deviation"])
μ_context = read(hfile_context["context mean"])
std_context = read(hfile_context["context standard deviation"])
samples = read(hfile_sample["samples"])
snapshots = read(hfile_sample["data"] )
ctrain = read(hfile_sample["context"])
inds = read(hfile_sample["context indices"])
=#
close(hfile_context)
close(hfile_sample)



cr = (quantile(snapshots[:], 1-qp), quantile(snapshots[:],qp))
fig = Figure(resolution = (900, 900))
random_indices = rand(1:size(snapshots)[end], N^2)
random_choices = rand(0:1, N^2)

fig3 = Figure(resolution = (900, 900))
N = 4
for i in 1:N
    ax = GLMakie.Axis(fig3[1, i]; title = "Field")
    GLMakie.heatmap!(ax, snapshots[:,:,1, 250 * i], colorrange = cr, colormap = colormap)
    ax = GLMakie.Axis(fig3[2, i]; title = "Context")
    GLMakie.heatmap!(ax, snapshots[:,:,2, 250 * i], colorrange = cr, colormap = colormap)
end
display(fig3)