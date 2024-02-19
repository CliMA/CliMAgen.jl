using GLMakie, HDF5, Random, Statistics
Random.seed!(1234)


hfile_sample = h5open("data/data_2.0_0.0_0.0_0.0_context_analysis.hdf5", "r")
hfile_context = h5open("data/data_2.0_0.0_0.0_0.0_context.hdf5", "r")

snapshots = read(hfile_context["snapshots"])
samples = read(hfile_sample["samples"])
xtrain = read(hfile_sample["data"] )
ctrain = read(hfile_sample["context"])
inds = read(hfile_sample["context indices"])
close(hfile_context)
close(hfile_sample)

N = 4
colormap = :balance
qp = 0.9

context_index = inds[1]
fig5 = Figure(resolution = (900, 900))
N = 4
skip = 4

cr = (quantile(xtrain[:], 1-qp), quantile(xtrain[:], qp))
i = 1
ii = (i-1)÷4 + 1
jj = (i-1)%4 + 1
k = i
ax = GLMakie.Axis(fig5[ii,jj]; title = "Conditional Information")
GLMakie.heatmap!(ax, ctrain[:, :, 1, context_index]' , colorrange = cr, colormap = colormap)

i = 2
ii = (i-1)÷4 + 1
jj = (i-1)%4 + 1
k = i
ax = GLMakie.Axis(fig5[ii,jj]; title = "True Field")
GLMakie.heatmap!(ax, xtrain[:, :, 1, context_index]', colorrange = cr, colormap = colormap)

for i in 3:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    k = i
    ax = GLMakie.Axis(fig5[ii,jj]; title = "Generated Sample")
    GLMakie.heatmap!(ax, samples[:,:, 1, i]', colorrange = cr, colormap = colormap)
end

display(fig5)