using GLMakie, HDF5, Random, Statistics
Random.seed!(1234)


hfile_sample = h5open("data/data_5.0_0.0_0.0_0.0_context_analysis.hdf5", "r")
hfile_context = h5open("data/data_5.0_0.0_0.0_0.0_context.hdf5", "r")

snapshots = read(hfile_context["snapshots"])
μ = read(hfile_context["mean"])
σ = read(hfile_context["standard deviation"])
μ_context = read(hfile_context["context mean"])
std_context = read(hfile_context["context standard deviation"])
samples = read(hfile_sample["samples"])
xtrain = read(hfile_sample["data"] )
ctrain = read(hfile_sample["context"])
inds = read(hfile_sample["context indices"])
close(hfile_context)
close(hfile_sample)

# snapshots[:,:, 1, :] = snapshots[:,:, 1, :] * σ .+ μ 
# snapshots[:,:, 2, :] = snapshots[:,:, 2, :] * std_context .+ μ_context
# samples = samples .* σ .+ μ

N = 4
colormap = :thermometer
qp = 0.95
#=
cr = (quantile(snapshots[:], 1-qp), quantile(snapshots[:],qp))
fig = Figure(resolution = (900, 900))
random_indices = rand(1:size(snapshots)[end], N^2)
random_choices = rand(0:1, N^2)
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
=#
##
#=
context_index = 1
fig5 = Figure(resolution = (900, 900))
N = 4
skip = 4

cr = (quantile(samples[:], 1-qp), quantile(samples[:],qp))
μ_shift = mean(samples[:,:, 1, :]) .- mean(snapshots[:,:,1, context_index])
i = 1
ii = (i-1)÷4 + 1
jj = (i-1)%4 + 1
k = i
ax = GLMakie.Axis(fig5[ii,jj]; title = "Conditional Information (color shift)")
GLMakie.heatmap!(ax, snapshots[:,:, 2, context_index] .+ μ_shift, colorrange = cr, colormap = colormap)

i = 2
ii = (i-1)÷4 + 1
jj = (i-1)%4 + 1
k = i
ax = GLMakie.Axis(fig5[ii,jj]; title = "True Field (color shift)")
GLMakie.heatmap!(ax, snapshots[:,:,1, context_index] .+ μ_shift, colorrange = cr, colormap = colormap)

for i in 3:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    k = i
    ax = GLMakie.Axis(fig5[ii,jj]; title = "Generated Sample")
    GLMakie.heatmap!(ax, samples[:,:, 1, i], colorrange = cr, colormap = colormap)
end

display(fig5)
=#

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
GLMakie.heatmap!(ax, ctrain[:, :, 1, context_index] , colorrange = cr, colormap = colormap)

i = 2
ii = (i-1)÷4 + 1
jj = (i-1)%4 + 1
k = i
ax = GLMakie.Axis(fig5[ii,jj]; title = "True Field")
GLMakie.heatmap!(ax, xtrain[:, :, 1, context_index], colorrange = cr, colormap = colormap)

for i in 3:N^2
    ii = (i-1)÷4 + 1
    jj = (i-1)%4 + 1
    k = i
    ax = GLMakie.Axis(fig5[ii,jj]; title = "Generated Sample")
    GLMakie.heatmap!(ax, samples[:,:, 1, i], colorrange = cr, colormap = colormap)
end

display(fig5)