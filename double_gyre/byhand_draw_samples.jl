include("sampler.jl")
using CairoMakie

nsamples = 100
nsteps = 250 
resolution = (128, 128)
time_steps, Δt, init_x = setup_sampler(
    score_model_smooth,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)


ntotal = 1000
total_samples = zeros(resolution..., inchannels, ntotal)
cprelim = zeros(resolution..., context_channels, nsamples)
rng = MersenneTwister(1234)
tot = ntotal ÷ nsamples
for i in ProgressBar(1:tot)
    if i ≤ (tot ÷ 2)
        # cprelim = reshape(gfp(Float32(0.1)), (192, 96, 1, 1)) * gfp_scale
        cprelim .= contextfield[:, :, :, 1:1] # ensemble_mean[:, :, :, 1:1]
    else
        # cprelim = reshape(gfp(Float32(1.0)), (192, 96, 1, 1)) * gfp_scale
        cprelim .= contextfield[:, :, :, end:end]# ensemble_mean[:, :, :, end:end]
    end
    c = device(cprelim)
    samples = Array(Euler_Maruyama_sampler(score_model_smooth, init_x, time_steps, Δt; rng, c))
    total_samples[:, :, :, (i-1)*nsamples+1:i*nsamples] .= samples
end


fig = Figure()
stateindex = 1
ax = Axis(fig[1, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, 1]; colormap = :viridis)
ax = Axis(fig[1, 2]; title = "ground truth")
crange = extrema(field[:, :, stateindex, N+1])
heatmap!(ax, field[:, :, stateindex, N+1], colormap = :viridis)
ax = Axis(fig[1, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = :viridis)
ax = Axis(fig[1, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = :viridis)

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = :viridis)
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, N+2], colormap = :viridis)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = :viridis)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = :viridis)

save("double_gyre_samples_u.png", fig)

fig = Figure()
stateindex = 2
ax = Axis(fig[1, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, 1]; colormap = :viridis)
ax = Axis(fig[1, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, N+1], colormap = :viridis)
ax = Axis(fig[1, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = :viridis)
ax = Axis(fig[1, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = :viridis)

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = :viridis)
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, N+2], colormap = :viridis)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = :viridis)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = :viridis)

save("double_gyre_samples_v.png", fig)

fig = Figure()
stateindex = 3
ax = Axis(fig[1, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, 1]; colormap = :viridis)
ax = Axis(fig[1, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, N+1], colormap = :viridis)
ax = Axis(fig[1, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = :viridis)
ax = Axis(fig[1, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = :viridis)

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = :viridis)
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, N+2], colormap = :viridis)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = :viridis)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = :viridis)

save("double_gyre_samples_b.png", fig)

fig = Figure()
ax = Axis(fig[1,1]; title = "losses")
lines!(ax, [loss[1] for loss in losses], color = :blue)
lines!(ax, [loss[1] for loss in losses_test], color = :red)
save("losses_double_gyre.png", fig)


#=
ax = Axis(fig[2, 1]; title = "context")
hist!(ax, contextfield[:, :, 1, 1][:])
ax = Axis(fig[2, 2]; title = "samples and truth")
hist!(ax, field[:, :, stateindex, N+1][:], color = (:blue, 0.5))
hist!(ax, total_samples[:, :, stateindex, 1][:], color = (:red, 0.5))
=#