include("sampler.jl")

score_model_smooth = load_model(checkpoint_path)
using CairoMakie

nsamples = 100
nsteps = 250 * 2
resolution = (192, 96)
time_steps, Δt, init_x = setup_sampler(
    score_model_smooth,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)


ntotal = 5000
total_samples = zeros(resolution..., inchannels, ntotal)
rng = MersenneTwister(1234)
for i in ProgressBar(1:50)
    samples = Array(Euler_Maruyama_sampler(score_model_smooth, init_x, time_steps, Δt; rng))
    total_samples[:, :, :, (i-1)*nsamples+1:i*nsamples] .= samples
end

colorrange = extrema(field)
fig = Figure()
ax = Axis(fig[1, 1]; title = "ai")
heatmap!(ax, Array(samples[:,:,1,1]); colorrange, colormap = :balance)
ax = Axis(fig[1, 2]; title = "ai")
hist!(ax, Array(samples[:,:,1,:])[:], bins = 100)
xlims!(ax, colorrange)
ax = Axis(fig[2, 1]; title = "data")
heatmap!(ax, Array(field[:,:,1,1]); colorrange, colormap = :balance)
ax = Axis(fig[2, 2]; title = "data")
hist!(ax, Array(field[:,:,1,:])[:], bins = 100)
xlims!(ax, colorrange)
save("samples.png", fig)


for i in 1:10
    fig = Figure()
    i1 = rand(1:192)# 150
    j1 = rand(1:96)# 48
    i2 = rand(1:192)# 48
    j2 = rand(1:96) # 48
    ax = Axis(fig[1, 1]; title = "ai ($i1,$j1)")
    hist!(ax, Array(total_samples[i1,j1,1,:])[:], bins = 20)
    ax = Axis(fig[1, 2]; title = "ai ($i2,$j2)")
    hist!(ax, Array(total_samples[i2,j2,1,:])[:], bins = 20)
    ax = Axis(fig[2, 1]; title = "data ($i1,$j1)")
    hist!(ax, Array(field[i1,j1,1,:])[:], bins = 20)
    ax = Axis(fig[2, 2]; title = "data ($i2,$j2)")
    hist!(ax, Array(field[i2,j2,1,:])[:], bins = 20)
    save("samples_hist_$i.png", fig)
end