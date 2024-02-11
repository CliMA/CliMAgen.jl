using GLMakie, HDF5, Random, Statistics
Random.seed!(1234)
hfile_true = h5open("data/data_0.0_0.0_0.0_0.0.hdf5", "r")
hfile_sample = h5open("data/data_0.0_0.0_0.0_0.0_analysis.hdf5", "r")


snapshots = read(hfile_true["snapshots"])
μ = read(hfile_true["mean"])
σ = read(hfile_true["standard deviation"])
samples = read(hfile_sample["samples"])
close(hfile_true)
close(hfile_sample)

snapshots = snapshots * σ .+ μ 
samples = samples .* σ .+ μ

qp = 0.95
cr = (quantile(samples[:], 1-qp), quantile(samples[:],qp))
fig = Figure(resolution = (900, 900))
random_indices = rand(1:size(snapshots)[end], N^2)
random_choices = rand(0:1, N^2)
N = 4
colormap = :afmhot
for i in 1:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = GLMakie.Axis(fig[jj, ii])
    if random_choices[i] == 0
        GLMakie.heatmap!(ax, snapshots[:,:,1, random_indices[i]], colorrange = cr, colormap = colormap)
    else
        GLMakie.heatmap!(ax, samples[:,:, 1, i], colorrange = cr, colormap = colormap)
    end
    hidedecorations!(ax)
end
display(fig)


fig2 = Figure(resolution = (900, 900))
N = 4
colormap = :afmhot
for i in 1:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    if random_choices[i] == 0
        ax = GLMakie.Axis(fig2[jj, ii]; title = "True")
        GLMakie.heatmap!(ax, snapshots[:,:,1, random_indices[i]], colorrange = cr, colormap = colormap)
    else
        ax = GLMakie.Axis(fig2[jj, ii]; title = "Generated")
        GLMakie.heatmap!(ax, samples[:,:, 1, i], colorrange = cr, colormap = colormap)
    end
    hidedecorations!(ax)
end
display(fig2)


fig3 = Figure(resolution = (900, 900))
N = 4
skip = 4
for i in 1:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    k = (i-1)*skip + 1
    ax = GLMakie.Axis(fig3[ii,jj]; title = "($k, $k)")
    hist!(ax, samples[k,k,1, :], color= (:red, 0.5), normalization = :pdf)
    hist!(ax, snapshots[k,k,1, :], color= (:blue, 0.5), normalization = :pdf)
end
display(fig3)

##
fig4 = Figure(resolution = (900, 900))
N = 4
skip = 4
for i in 1:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    k = (i-1)*skip + 1
    ax = GLMakie.Axis(fig3[ii,jj]; title = "($k, $k)")
    hist!(ax, samples[k,k,1, :], color= (:red, 0.5), normalization = :pdf)
    hist!(ax, snapshots[k,k,1, :], color= (:blue, 0.5), normalization = :pdf)
end
display(fig4)

averaged_samples = mean([samples[i:skip:end, j:skip:end, 1, :] for i in 1:skip, j in 1:skip])[:, :, :]
averaged_snapshots = mean([snapshots[i:skip:end, j:skip:end, 1, :] for i in 1:skip, j in 1:skip])[:, :, :]

fig4 = Figure(resolution = (900, 900))
N = 4
skip = 4
for i in 1:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    k = i
    ax = GLMakie.Axis(fig4[ii,jj]; title = "($k, $k)")
    hist!(ax, averaged_samples[k,k, :], color= (:red, 0.5), normalization = :pdf)
    hist!(ax, averaged_snapshots[k,k, :], color= (:blue, 0.5), normalization = :pdf)
end
display(fig4)