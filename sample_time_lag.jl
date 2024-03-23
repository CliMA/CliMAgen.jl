using GLMakie, Random
nsamples = 8
nsteps = 250
Ny = 64
resolution = (Ny, Ny)

rng = MersenneTwister(1234)

ctrain = zeros(Float32, Ny, Ny, 2, nsamples)

rbatch = copy(reshape(gated_array, (128, 64, 2, batchsize)));
batch = (rbatch[1:2:end, :, :, :] + rbatch[2:2:end, :, :, :]) / (2σ)
ctrain[:, :, 1, :] .= Float32.(batch[:, :, 2, [8]])
ctrain[:, :, 2, :] .= cpu_batch[:, :, 3, [8]]

time_steps, Δt, init_x = setup_sampler(
    score_model_smooth,
    device,
    resolution[1],
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)

samples = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
samples = cpu(samples)


colorrange = (-1.0, 1.0)
fig = Figure()
ax = Axis(fig[1, 1]; title = "ai next timestep")
heatmap!(ax, samples[:,:,1,1]; colorrange, colormap = :balance)
ax = Axis(fig[1, 2]; title = "ai next timestep")
heatmap!(ax, samples[:,:,1,8]; colorrange, colormap = :balance)
ax = Axis(fig[2, 1]; title = "training data current timestep")
heatmap!(ax, batch[:,:,2,8]; colorrange, colormap = :balance)
ax = Axis(fig[2, 2]; title = "training data next timestep")
heatmap!(ax, batch[:,:,1,8]; colorrange, colormap = :balance)
display(fig)

fig2 = Figure() 
ax = Axis(fig2[1,1]; title = "context")
heatmap!(ax, batch[:, :, 2, 8]; colorrange, colormap = :balance)
ax = Axis(fig2[1,2]; title = "truth")
heatmap!(ax, batch[:, :, 1, 8]; colorrange, colormap = :balance)
for i in 3:9
    ii = (i-1)÷3 + 1
    jj = (i-1)%3 + 1
    ax = Axis(fig2[ii, jj]; title = "ai")
    heatmap!(ax, samples[:,:,1,i-2]; colorrange, colormap = :balance)
end
display(fig2)

#=
xtrain = train[:, :, 1:noised_channels, tmp_shuffle]
old_ctrain = train[:, :, (noised_channels+1):(noised_channels+context_channels), tmp_shuffle]
ctrain = Float32.(copy(old_ctrain))

rng = MersenneTwister(1234)

tlag = 0
t = tlag/60
τ0 = reshape(gfp(t), (size(xtrain)[1], size(xtrain)[2], 1, 1))
ctrain[:, :, [2], 1:nsamples] .= τ0

nsteps = 10
us = zeros(Float32, 32, 32, nsteps);
us[:, :, 1] .= ctrain[:, :, [1], 1:nsamples]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

=#