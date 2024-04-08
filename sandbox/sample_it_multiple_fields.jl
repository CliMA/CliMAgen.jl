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

p = 0.05
rbatch = copy(reshape(gated_array, (128, 64, length(my_fields), batchsize)))
batch = (rbatch .- reshape(μ, (1, 1, length(my_fields), 1))) ./ reshape(σ, (1, 1, length(my_fields), 1))


M = length(fields)
N = length(layers)

# for j in 1:N
    # for i in 1:M
        # ii = (i-1)*N + j
        fig = Figure(resolution = (1200, 600))
        ax = Axis(fig[1, 1]; title = "ai: T")
        colorrange1 = (quantile(Array(samples[:,:,1,1])[:], p),  quantile(Array(samples[:,:,1,1])[:], 1-p))
        heatmap!(ax, lon, lat, Array(samples)[:,:,1,1]; colorrange1, colormap = :thermometer)
        ax = Axis(fig[1, 2]; title = "training data: T")
        heatmap!(ax, lon, lat, Array(batch[:,:,1,1]); colorrange1, colormap = :thermometer)

        ax = Axis(fig[1, 1]; title = "ai: T")
        colorrange1 = (quantile(Array(samples[:,:,1,1])[:], p),  quantile(Array(samples[:,:,1,1])[:], 1-p))
        heatmap!(ax, lon, lat, Array(samples)[:,:,1,1]; colorrange1, colormap = :thermometer)
        ax = Axis(fig[1, 2]; title = "training data: T")
        heatmap!(ax, lon, lat, Array(batch[:,:,1,1]); colorrange1, colormap = :thermometer)

        ax = Axis(fig[1, 1+2]; title = "ai: ω")
        ind = 2
        colorrange1 = (quantile(Array(samples[:,:,ind,1])[:], p),  -quantile(Array(samples[:,:,ind,1])[:], p),)
        heatmap!(ax, lon, lat, Array(samples)[:,:,ind,1]; colorrange1, colormap = :balance)
        ax = Axis(fig[1, 2+2]; title = "training data: ω")
        heatmap!(ax, lon, lat, Array(batch[:,:,ind,1]); colorrange1, colormap = :balance)

        ax = Axis(fig[2, 1]; title = "ai: humidity")
        ind = 3
        colorrange1 = (quantile(Array(samples[:,:,ind,1])[:], p),  quantile(Array(samples[:,:,ind,1])[:], 1-p))
        heatmap!(ax, lon, lat, Array(samples)[:,:,ind,1]; colorrange1, colormap = :blues)
        ax = Axis(fig[2, 2]; title = "training data: humidity")
        heatmap!(ax, lon, lat, Array(batch[:,:,ind,1]); colorrange1, colormap = :blues)

        ax = Axis(fig[2, 1+2]; title = "ai: div")
        ind = 4
        colorrange1 = (quantile(Array(samples[:,:,ind,1])[:], p),  -quantile(Array(samples[:,:,ind,1])[:], p))
        heatmap!(ax, lon, lat, Array(samples)[:,:,ind,1]; colorrange1, colormap = :balance)
        ax = Axis(fig[2, 2+2]; title = "training data: div")
        heatmap!(ax, lon, lat, Array(batch[:,:,ind,1]); colorrange1, colormap = :balance)
        save("after_training_multipe_fields_layer_$j.png", fig)
    # end
# end
display(fig)

