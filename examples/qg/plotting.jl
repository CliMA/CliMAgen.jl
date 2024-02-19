using GLMakie, HDF5, Random, Statistics
Random.seed!(1234)
hfile_true = h5open("data/data_0.0_0.0_0.0_0.0.hdf5", "r")
hfile_sample = h5open("data/data_0.0_0.0_0.0_0.0_analysis.hdf5", "r")


snapshots = read(hfile_true["snapshots"])
samples = read(hfile_sample["samples"])
close(hfile_true)
close(hfile_sample)


println(std(samples), " vs ", std(snapshots))

N = 4
colormap = :balance
qp = 0.99

field_ind = 2
cr2 = (quantile(snapshots[:, :, field_ind, :][:], 1-qp), quantile(snapshots[:, :, field_ind, :][:],qp))

field_ind = 1
cr = (quantile(snapshots[:, :, field_ind, :][:], 1-qp), quantile(snapshots[:, :, field_ind, :][:],qp))
fig = Figure(resolution = (900, 900))
random_indices = rand(1:size(snapshots)[end], N^2)
random_choices = rand(0:1, N^2)
for i in 1:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    ax = GLMakie.Axis(fig[jj, ii])
    if random_choices[i] == 0
        GLMakie.heatmap!(ax, snapshots[:,:,field_ind, random_indices[i]]', colorrange = cr, colormap = colormap)
    else
        GLMakie.heatmap!(ax, samples[:,:, field_ind, i]', colorrange = cr, colormap = colormap)
    end
    hidedecorations!(ax)
end
display(fig)


fig2 = Figure(resolution = (900, 900))
N = 4
for i in 1:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    if random_choices[i] == 0
        ax = GLMakie.Axis(fig2[jj, ii]; title = "True")
        GLMakie.heatmap!(ax, snapshots[:,:,field_ind, random_indices[i]]', colorrange = cr, colormap = colormap)
    else
        ax = GLMakie.Axis(fig2[jj, ii]; title = "Generated")
        GLMakie.heatmap!(ax, samples[:,:, field_ind, i]', colorrange = cr, colormap = colormap)
    end
    hidedecorations!(ax)
end
display(fig2)


fig3 = Figure(resolution = (900, 900))
N = 4
for i in 1:N
    ax = GLMakie.Axis(fig3[1, i]; title = "Generated q1")
    GLMakie.heatmap!(ax, samples[:,:, 1, i]', colorrange = cr, colormap = colormap)
    hidedecorations!(ax)
    ax = GLMakie.Axis(fig3[2, i]; title = "Generated q2")
    GLMakie.heatmap!(ax, samples[:,:, 2, i]', colorrange = cr2, colormap = colormap)
    hidedecorations!(ax)

    ax = GLMakie.Axis(fig3[3, i]; title = "Data q1")
    GLMakie.heatmap!(ax, snapshots[:,:,1,random_indices[i]]', colorrange = cr, colormap = colormap)
    hidedecorations!(ax)
    ax = GLMakie.Axis(fig3[4, i]; title = "Data q2")
    GLMakie.heatmap!(ax, snapshots[:,:,2,random_indices[i]]', colorrange = cr2, colormap = colormap)
    hidedecorations!(ax)
end
display(fig3)
