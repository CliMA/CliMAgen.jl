using GLMakie
using HDF5, Random, Statistics
Random.seed!(1234)


hfile_sample = h5open("data/data_4.0_0.0_0.0_0.0_context_analysis.hdf5", "r")
hfile_context = h5open("data/data_4.0_0.0_0.0_0.0_context.hdf5", "r")

snapshots = read(hfile_context["snapshots"])
samples = read(hfile_sample["samples"])
xtrain = read(hfile_sample["data"] )
ctrain = read(hfile_sample["context"])
ctrains = read(hfile_sample["contexts"])
inds = read(hfile_sample["context indices"])
μs = read(hfile_context["mean"])
σs = read(hfile_context["standard deviation"])
samples_c = read(hfile_sample["samples with various conditionals"])
close(hfile_context)
close(hfile_sample)


M = size(xtrain)[1]
N = 4
colormap = :balance
qp = 0.99

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

##
fig4 = Figure() 

ax = GLMakie.Axis(fig4[1, 1]; title = "True Field")
GLMakie.heatmap!(ax, xtrain[:, :, 1, context_index]', colorrange = cr, colormap = colormap)

ax = GLMakie.Axis(fig4[1, 2]; title = "Sample")
GLMakie.heatmap!(ax, samples[:,:, 1, 1]', colorrange = cr, colormap = colormap)

ax = GLMakie.Axis(fig4[1, 3]; title = "Mean")
GLMakie.heatmap!(ax, mean(samples, dims = 4)[:, :, 1, 1]', colorrange = cr, colormap = colormap)

ax = GLMakie.Axis(fig4[1, 4]; title = "Standard Devation")
GLMakie.heatmap!(ax, std(samples, dims= 4)[:, :, 1, 1]', colorrange = cr, colormap = colormap)
display(fig4)

##
fig3 = Figure() 

p1x_e = ctrain[:, :, [1], [context_index]] - circshift(ctrain[:, :, [1], [context_index]], (0, 2, 0, 0))
p2_e = samples[:, :, [1], :]

p1x = ctrain[:, :, 1, context_index] - circshift(ctrain[:, :, 1, context_index], (0, 2))
p2 = xtrain[:, :, 1, context_index]
p2_s = samples[:,:, 1, 1]
p2_m = mean(samples, dims = 4)[:, :, 1, 1]

heatflux1 = mean(p1x .* p2, dims = 2)[:] 
heatflux2 = mean(p1x .* p2_s, dims = 2)[:]
heatflux3 = mean(p1x .* p2_m, dims = 2)[:]

heatflux4 = mean(p1x_e .* p2_e, dims = (2,3,4))[:]
heatflux_prep = mean(p1x_e .* p2_e, dims = (2, 3))[:, 1, 1, :]
heatflux_std = 5 * std(heatflux_prep, dims = 2)[:]
heatflux_upper = maximum(heatflux_prep, dims = 2)[:]
heatflux_lower = minimum(heatflux_prep, dims = 2)[:]
qu = 1.0
heatflux_upper_quantile = [quantile(heatflux_prep[i, :], qu) for i in 1:M] - heatflux4
heatflux_lower_quantile = heatflux4 - [quantile(heatflux_prep[i, :], 1-qu) for i in 1:M]

lats = collect(1:M)

options = (; xlabel = "Heat Flux", ylabel = "Latitude")
ax = GLMakie.Axis(fig3[1, 1]; title = "True Field: Heat Flux", options...)
GLMakie.lines!(ax, heatflux1, lats, color = :blue)

ax = GLMakie.Axis(fig3[1, 2]; title = "Sample: Heat Flux", options...)
GLMakie.lines!(ax, heatflux2, lats, color = :red)

ax = GLMakie.Axis(fig3[2, 1]; title = "Generative: Heat Flux", options...)
# GLMakie.lines!(ax, heatflux4, lats, color = :orange)
GLMakie.scatter!(ax, heatflux4, lats, color = :orange)
errorbars!(ax, heatflux4, lats, heatflux_lower_quantile, heatflux_upper_quantile, whiskerwidth = 3, direction = :x, color = :orange)

ax = GLMakie.Axis(fig3[2, 2]; title = "Heat Flux: Together", options...)
GLMakie.lines!(ax, heatflux1, lats, color = :blue)
GLMakie.lines!(ax, heatflux2, lats, color = :red)
# GLMakie.lines!(ax, heatflux4, lats, color = :orange)
GLMakie.scatter!(ax, heatflux4, lats, color = :orange)
errorbars!(ax, heatflux4, lats, heatflux_lower_quantile, heatflux_upper_quantile,  whiskerwidth = 3, direction = :x, color = (:orange, 0.5))

display(fig3)


##

p1x_e = ctrains[:, :, [1], :] - circshift(ctrains[:, :, [1], :], (0, 2, 0, 0))
p2_e  = samples_c[:, :, [1], :]

p2 = xtrain[:, :, [1], :]

Ne = size(samples_c)[end]
heatflux1 = mean(p1x_e[:, :, [1], 1:Ne] .* p2[:, :, [1], 1:Ne], dims = (2,3,4))[:]
heatflux4 = mean(p1x_e[:, :, [1], 1:Ne] .* p2_e, dims = (2,3,4))[:]
#=
heatflux_prep = mean(p1x_e .* p2_e, dims = (2, 3))[:, 1, 1, :]
heatflux_std = 5 * std(heatflux_prep, dimshp = 2)[:]
heatflux_upper = maximum(heatflux_prep, dims = 2)[:]
heatflux_lower = minimum(heatflux_prep, dims = 2)[:]
qu = 1.0
heatflux_upper_quantile = [quantile(heatflux_prep[i, :], qu) for i in 1:M] - heatflux4
heatflux_lower_quantile = heatflux4 - [quantile(heatflux_prep[i, :], 1-qu) for i in 1:M]
=#

fig2 = Figure() 
lats = collect(1:M)
options = (; xlabel = "Heat Flux", ylabel = "Latitude")
ax = GLMakie.Axis(fig2[1, 1]; title = "Generated Field: Heat Flux", options...)
GLMakie.lines!(ax, heatflux4, lats, color = :blue)

ax = GLMakie.Axis(fig2[1, 2]; title = "True Field: Heat Flux", options...)
GLMakie.lines!(ax, heatflux1, lats, color = :orange)

ax = GLMakie.Axis(fig2[1, 3]; title = "Together Field: Heat Flux", options...)
GLMakie.lines!(ax, heatflux4, lats, color = :blue)
GLMakie.lines!(ax, heatflux1, lats, color = :orange)

display(fig2)

##
skip = 10
fig1 = Figure()
for i in 1:4
    ax = GLMakie.Axis(fig1[1, i]; title = "Context $i")
    GLMakie.heatmap!(ax, ctrains[:,:, 1, i*skip]', colorrange = cr, colormap = colormap)
end

for i in 1:4
    ax = GLMakie.Axis(fig1[3, i]; title = "Generated $i")
    GLMakie.heatmap!(ax, samples_c[:,:, 1, i*skip]', colorrange = cr, colormap = colormap)
end
for i in 1:4
    ax = GLMakie.Axis(fig1[2, i]; title = "Truth $i")
    GLMakie.heatmap!(ax, p2[:,:, 1, i*skip]', colorrange = cr, colormap = colormap)
end
display(fig1)

