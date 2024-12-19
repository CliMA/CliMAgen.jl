include("sampler.jl")
using CairoMakie
# /orcd/data/raffaele/001/sandre/OceananigansData
nsamples = 10
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


ntotal = 20
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
ax = Axis(fig[1,1]; title = "losses")
lines!(ax, [loss[1] for loss in losses], color = :blue)
lines!(ax, [loss[1] for loss in losses_test], color = :red)
save("losses_double_gyre_$casevar.png", fig)


fig = Figure(resolution = (1600, 800))
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

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)

save("double_gyre_samples_u_case_$casevar.png", fig)

fig = Figure(resolution = (1600, 800))
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

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)

save("double_gyre_samples_v_$casevar.png", fig)

fig = Figure(resolution = (1600, 800))
stateindex = 3
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

ax = Axis(fig[2, 1]; title = "context")
heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
ax = Axis(fig[2, 2]; title = "ground truth")
heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 3]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
ax = Axis(fig[2, 4]; title = "samples")
heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)

save("double_gyre_samples_b_$casevar.png", fig)




#=
ax = Axis(fig[2, 1]; title = "context")
hist!(ax, contextfield[:, :, 1, 1][:])
ax = Axis(fig[2, 2]; title = "samples and truth")
hist!(ax, field[:, :, stateindex, contextind1][:], color = (:blue, 0.5))
hist!(ax, total_samples[:, :, stateindex, 1][:], color = (:red, 0.5))
=#