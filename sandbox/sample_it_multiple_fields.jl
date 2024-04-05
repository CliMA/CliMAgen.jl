include("sampler.jl")
using CairoMakie
nsamples = 1
nsteps = 250
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


lat, lon = RingGrids.get_latdlonds(my_fields[1].var)
lon = reshape(lon, (128, 64))[:,1]
lat = reshape(lat, (128, 64))[1, :]

colorrange = (-1.0, 1.0)
fig = Figure()
ax = Axis(fig[1, 1]; title = "ai")
heatmap!(ax, lon, lat, Array(samples)[:,:,1,1]; colorrange, colormap = :balance)
ax = Axis(fig[1, 2]; title = "training data")
rbatch = copy(reshape(gated_array, (128, 64, length(my_fields), batchsize)))
batch = (rbatch .- reshape(μ, (1, 1, length(my_fields), 1))) ./ reshape(σ, (1, 1, length(my_fields), 1))
heatmap!(ax, lon, lat, Array(batch[:,:,1,1]); colorrange, colormap = :balance)
display(fig)

save("after_training_multipe_fields.png", fig)