include("sampler.jl")
using CairoMakie

nsamples = 100
nsteps = 250 
resolution = (192, 96)
time_steps, Δt, init_x = setup_sampler(
    score_model_smooth,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)


ntotal = 10000
total_samples = zeros(resolution..., inchannels, ntotal)
rng = MersenneTwister(1234)
c = copy(init_x)
tot = ntotal ÷ nsamples
for i in ProgressBar(1:tot)
    if i ≤ (tot ÷ 2)
        # cprelim = reshape(gfp(Float32(0.1)), (192, 96, 1, 1)) * gfp_scale
        cprelim = contextfield[:, :, :, 1:1] # ensemble_mean[:, :, :, 1:1]
    else
        # cprelim = reshape(gfp(Float32(1.0)), (192, 96, 1, 1)) * gfp_scale
        cprelim = contextfield[:, :, :, end:end]# ensemble_mean[:, :, :, end:end]
    end
    c .= device(cprelim)
    samples = Array(Euler_Maruyama_sampler(score_model_smooth, init_x, time_steps, Δt; rng, c))
    total_samples[:, :, :, (i-1)*nsamples+1:i*nsamples] .= samples
end

colorrange = extrema(field)
fig = Figure()
ax = Axis(fig[1, 1]; title = "ai")
heatmap!(ax, Array(total_samples[:,:,1,1]); colorrange, colormap = :balance)
ax = Axis(fig[1, 2]; title = "ai")
hist!(ax, Array(total_samples[:,:,1,1:ntotal÷2])[:], bins = 100)
xlims!(ax, colorrange)
ax = Axis(fig[2, 1]; title = "data")
heatmap!(ax, Array(total_samples[:,:,1,end]); colorrange, colormap = :balance)
ax = Axis(fig[2, 2]; title = "data")
hist!(ax, Array(total_samples[:,:,1,ntotal÷2+1:end])[:], bins = 100)
xlims!(ax, colorrange)
save("samples_pr_2.png", fig)

index_1 = [59, 90]
index_2 = [130, 46]
index_3 = [140, 47]
total_samples_physical = (total_samples .* physical_sigma) .+ physical_mu

fig  = Figure(resolution = (1200, 400))
binsize = 30
ax = Axis(fig[1, 1]; title = "ai ($index_1)")
hist!(ax, Array(total_samples_physical[index_1[1],index_1[2],1,1:ntotal÷2])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(total_samples_physical[index_1[1],index_1[2],1,(ntotal÷2+1):end])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
ax = Axis(fig[1, 2]; title = "ai ($index_2)")
hist!(ax, Array(total_samples_physical[index_2[1],index_2[2],1,1:ntotal÷2])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(total_samples_physical[index_2[1],index_2[2],1,(ntotal÷2+1):end])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
ax = Axis(fig[1, 3]; title = "ai ($index_3)")
hist!(ax, Array(total_samples_physical[index_3[1],index_3[2],1,1:ntotal÷2])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(total_samples_physical[index_3[1],index_3[2],1,(ntotal÷2+1):end])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
save("samples_hist_pr_exp2.png", fig)

fig  = Figure(resolution = (1200, 800))
binsize = 30
ax = Axis(fig[1, 1]; title = "ai ($index_1)")
hist!(ax, Array(total_samples_physical[index_1[1],index_1[2],1,1:ntotal÷2])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(total_samples_physical[index_1[1],index_1[2],1,(ntotal÷2+1):end])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax, 295, 310)
ax = Axis(fig[1, 2]; title = "ai ($index_2)")
hist!(ax, Array(total_samples_physical[index_2[1],index_2[2],1,1:ntotal÷2])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(total_samples_physical[index_2[1],index_2[2],1,(ntotal÷2+1):end])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax, 296, 303)
ax = Axis(fig[1, 3]; title = "ai ($index_3)")
hist!(ax, Array(total_samples_physical[index_3[1],index_3[2],1,1:ntotal÷2])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(total_samples_physical[index_3[1],index_3[2],1,(ntotal÷2+1):end])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax,  260, 280)

n = 2
ax = Axis(fig[2, 1]; title = "data ($index_1)")
hist!(ax, Array(rfield[index_1[1],index_1[2],1, 35-n:35+n, :])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(rfield[index_1[1],index_1[2],1, end-5:end, :])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax, 295, 310)
ax = Axis(fig[2, 2]; title = "data ($index_2)")
hist!(ax, Array(rfield[index_2[1],index_2[2],1, 35-n:35+n, :])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(rfield[index_2[1],index_2[2],1, end-5:end, :])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax,  296, 303)
ax = Axis(fig[2, 3]; title = "data ($index_3)")
hist!(ax, Array(rfield[index_3[1],index_3[2],1,35-n:35+n, :])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(rfield[index_3[1],index_3[2],1, end-5:end, :])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax, 260, 280)
save("samples_hist_pr_exp2.png", fig)

fig = Figure() 
ax = Axis(fig[1, 1]; title = "losses")
lines!(ax, [loss[1] for loss in losses], label = "train")
lines!(ax, [loss[1] for loss in losses_test], label = "test")
save("losses_pr_exp2.png", fig)