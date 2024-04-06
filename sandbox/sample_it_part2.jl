include("sampler.jl")
using CairoMakie
nsamples = 1
nsteps = 250
Ny = 64
resolution = (128, 64)
time_steps, Δt, init_x = setup_sampler(
    score_model_smooth,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)

samples = Euler_Maruyama_sampler(score_model_smooth, init_x, time_steps, Δt)


fig = Figure()
p = 0.05
colorrange1 = (quantile(Array(samples[:,:,1,1])[:], p),  quantile(Array(samples[:,:,1,1])[:], 1-p))
ax = Axis(fig[1, 1]; title = "ai")
heatmap!(ax, Array(samples[:,:,1,1]); colorrange1, colormap = :thermometer)
ax = Axis(fig[1, 2]; title = "training data")
rbatch = copy(reshape(gated_array, (128, 64, 1, batchsize)))
batch = rbatch / (2σ)
heatmap!(ax, Array(batch[:,:,1,1]); colorrange, colormap = :thermometer)
display(fig)

save("after_training.png", fig)