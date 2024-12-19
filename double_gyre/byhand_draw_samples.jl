include("sampler.jl")

using CairoMakie
# /orcd/data/raffaele/001/sandre/OceananigansData
nsamples = 100
nsteps = 300 
resolution = size(field)[1:2]
time_steps, Δt, init_x = setup_sampler(
    score_model_smooth,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)


ntotal = nsamples * 2
total_samples = zeros(resolution..., inchannels, ntotal)
cprelim = zeros(resolution..., context_channels, nsamples)
rng = MersenneTwister(12345)
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

averaged_samples_1 = mean(total_samples[:,:, :, 1:nsamples], dims = 4)[:, :, :, 1]
std_samples_1 = std(total_samples[:,:, :, 1:nsamples], dims = 4)[:, :, :, 1]
averaged_samples_2 = mean(total_samples[:,:, :, nsamples+1:2 * nsamples], dims = 4)[:, :, :, 1]
std_samples_2 = std(total_samples[:,:, :, nsamples+1:2 * nsamples], dims = 4)[:, :, :, 1]


hfile = h5open(save_directory * prefix * "generative_samples.hdf5", "w" )
hfile["samples context 1"] = total_samples[:, :, :, 1:nsamples]
hfile["samples context 2"] = total_samples[:, :, :, nsamples+1:2 * nsamples]
hfile["context field 1"] = contextfield[:, :, :, 1:1]
hfile["context field 2"] = contextfield[:, :, :, end:end]
hfile["averaged samples 1"] = averaged_samples_1
hfile["std samples 1"] = std_samples_1
hfile["averaged samples 2"] = averaged_samples_2
hfile["std samples 2"] = std_samples_2
hfile["last training index"] = N
hfile["sample index 1"] = contextind1
hfile["sample index 2"] = contextind2
close(hfile)

fig = Figure()
ax = Axis(fig[1,1]; title = "losses")
lines!(ax, [loss[1] for loss in losses], color = :blue)
lines!(ax, [loss[1] for loss in losses_test], color = :red)
save(figure_directory  * "losses_double_gyre_$casevar.png", fig)


fig = Figure(resolution = (2400, 800))
stateindex = 1
ηmax = maximum(contextfield[:, :, 1, 1])
ηrange = (-ηmax, ηmax)
colormap_η = :balance
colormap = :balance
ax = Axis(fig[1, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, 1]; colormap = colormap_η, colorrange = ηrange  )
quval = quantile(field[:, :, stateindex, contextind1][:], 0.95)
crange = (-quval, quval)
ax = Axis(fig[1, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind1], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 5]; title = "mean")
heatmap!(ax, averaged_samples_1[:, :, stateindex], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 6]; title = "std")
heatmap!(ax, std_samples_1[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 5]; title = "mean")
heatmap!(ax, averaged_samples_2[:, :, stateindex], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 6]; title = "std")
heatmap!(ax, std_samples_2[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

save(figure_directory  * "double_gyre_samples_u_case_$casevar.png", fig)

fig = Figure(resolution = (2400, 800))
stateindex = 2
colormap = :balance
ax = Axis(fig[1, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, 1]; colormap = colormap_η, colorrange = ηrange  )
quval = quantile(field[:, :, stateindex, contextind1][:], 0.95)
crange = (-quval, quval)
ax = Axis(fig[1, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind1], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 5]; title = "mean")
heatmap!(ax, averaged_samples_1[:, :, stateindex], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 6]; title = "std")
heatmap!(ax, std_samples_1[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 5]; title = "mean")
heatmap!(ax, averaged_samples_2[:, :, stateindex], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 6]; title = "std")
heatmap!(ax, std_samples_2[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

save(figure_directory  * "double_gyre_samples_v_$casevar.png", fig)

fig = Figure(resolution=(2400, 800))
stateindex = 3
colormap = :balance
ax = Axis(fig[1, 1]; title="context")
heatmap!(ax, contextfield[:, :, 1, 1]; colormap=colormap_η, colorrange=ηrange)
quval = quantile(field[:, :, stateindex, contextind1][:], 0.95)
crange = (-quval, quval)
ax = Axis(fig[1, 2]; title="ground truth")
heatmap!(ax, field[:, :, stateindex, contextind1], colormap=colormap, colorrange=crange)
ax = Axis(fig[1, 3]; title="samples")
heatmap!(ax, total_samples[:, :, stateindex, 1], colormap=colormap, colorrange=crange)
ax = Axis(fig[1, 4]; title="samples")
heatmap!(ax, total_samples[:, :, stateindex, 2], colormap=colormap, colorrange=crange)
ax = Axis(fig[1, 5]; title = "mean")
heatmap!(ax, averaged_samples_1[:, :, stateindex], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 6]; title = "std")
heatmap!(ax, std_samples_1[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))


ax = Axis(fig[2, 1]; title="context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap=colormap_η, colorrange=ηrange)
ax = Axis(fig[2, 2]; title="ground truth")
heatmap!(ax, field[:, :, stateindex, contextind2], colormap=colormap, colorrange=crange)
ax = Axis(fig[2, 3]; title="samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap=colormap, colorrange=crange)
ax = Axis(fig[2, 4]; title="samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap=colormap, colorrange=crange)
ax = Axis(fig[2, 5]; title = "mean")
heatmap!(ax, averaged_samples_2[:, :, stateindex], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 6]; title = "std")
heatmap!(ax, std_samples_2[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

save(figure_directory  * "double_gyre_samples_w_$casevar.png", fig)

fig = Figure(resolution = (2400, 800))
stateindex = 4
colormap = :thermometer
ax = Axis(fig[1, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, 1]; colormap = colormap_η, colorrange = ηrange  )
quval = quantile(field[:, :, stateindex, contextind1][:], [0.05, 0.95])
crange = quval
ax = Axis(fig[1, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind1], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 5]; title = "mean")
heatmap!(ax, averaged_samples_1[:, :, stateindex], colormap = colormap, colorrange = crange)
ax = Axis(fig[1, 6]; title = "std")
heatmap!(ax, std_samples_1[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 5]; title = "mean")
heatmap!(ax, averaged_samples_2[:, :, stateindex], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 6]; title = "std")
heatmap!(ax, std_samples_2[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

save(figure_directory  * "double_gyre_samples_b_$casevar.png", fig)



stdfield = std(field[:, :, :, 1:N], dims = 4)
meanfield = mean(field[:, :, :, 1:N], dims = 4)

fig = Figure(resolution = (2400, 800))
for i in 1:5
    crange = (-quantile(meanfield[:, :, i, 1][:], 0.95), quantile(meanfield[:, :, i, 1][:], 0.95))
    ax = Axis(fig[1, i]; title = "mean $i")
    heatmap!(ax, meanfield[:, :, i, 1], colormap = :balance, colorrange = crange)
    ax = Axis(fig[2, i]; title = "std $i")
    heatmap!(ax, stdfield[:, :, i, 1], colormap = :viridis, colorrange = (0, quantile(stdfield[:, :, i, 1][:], 0.95)))
end

save(figure_directory  * "double_gyre_mean_std_$casevar.png", fig)
