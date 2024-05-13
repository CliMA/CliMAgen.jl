include("sampler.jl")
using CairoMakie
nsamples = 10
nsteps = 250 * 2
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
for j in 1:N
    fig = Figure(resolution = (1200, 300))
    ind = 1 + (j-1) * M
    colorrange1 = (quantile(Array(samples[:,:,ind,1])[:], p),  quantile(Array(samples[:,:,ind,1])[:], 1-p))
    ax = Axis(fig[1, 1]; title = "layer $j, training data: T")
    heatmap!(ax, lon, lat, Array(batch[:,:,ind,1]); colorrange1, colormap = :thermometer)
    for k in 1:4
        ax = Axis(fig[1, 1 + k]; title = "layer $j, ai: T, sample $k")
        heatmap!(ax, lon, lat, Array(samples)[:,:,ind,k]; colorrange1, colormap = :thermometer)
    end
    #=
    ax = Axis(fig[1, 1+2]; title = "layer $j, ai: ω")
    ind = 2 + (j-1) * M
    colorrange1 = (quantile(Array(samples[:,:,ind,1])[:], p),  -quantile(Array(samples[:,:,ind,1])[:], p),)
    heatmap!(ax, lon, lat, Array(samples)[:,:,ind,1]; colorrange1, colormap = :balance)
    ax = Axis(fig[1, 2+2]; title = "layer $j, training data: ω")
    heatmap!(ax, lon, lat, Array(batch[:,:,ind,1]); colorrange1, colormap = :balance)

    ax = Axis(fig[2, 1]; title = "layer $j, ai: humidity")
    ind = 3 + (j-1) * M
    colorrange1 = (quantile(Array(samples[:,:,ind,1])[:], p),  quantile(Array(samples[:,:,ind,1])[:], 1-p))
    heatmap!(ax, lon, lat, Array(samples)[:,:,ind,1]; colorrange1, colormap = :blues)
    ax = Axis(fig[2, 2]; title = "layer $j, training data: humidity")
    heatmap!(ax, lon, lat, Array(batch[:,:,ind,1]); colorrange1, colormap = :blues)

    ax = Axis(fig[2, 1+2]; title = "layer $j, ai: div")
    ind = 4 + (j-1) * M
    colorrange1 = (quantile(Array(samples[:,:,ind,1])[:], p),  -quantile(Array(samples[:,:,ind,1])[:], p))
    heatmap!(ax, lon, lat, Array(samples)[:,:,ind,1]; colorrange1, colormap = :balance)
    ax = Axis(fig[2, 2+2]; title = "layer $j, training data: div")
    heatmap!(ax, lon, lat, Array(batch[:,:,ind,1]); colorrange1, colormap = :balance)
    =#
    save("after_training_multipe_fields_layer_$j.png", fig)
end
    # end
# end

fig = Figure()
ax = Axis(fig[1, 1]; title = "losses", xlabel ="epoch /100", ylabel = "loss")
lines!(ax, losses, color = :red, label = "loss 1")
scatter!(ax, losses_2, color = (:blue, 0.25), label = "loss 2")
axislegend(ax, position = :rt)
save("losses.png", fig)

##
#=
fig2 = Figure(resolution = (1200, 600))
j = 2
ax = Axis(fig2[1, 1]; title = "ai: T")
colorrange1 = (quantile(Array(samples[:,:,1 + (j-1) * M,1])[:], p),  quantile(Array(samples[:,:,1 + (j-1) * M,1])[:], 1-p))
heatmap!(ax, lon, lat, Array(samples)[:,:,1 + (j-1) * M,1]; colorrange1, colormap = :thermometer)
ax = Axis(fig2[1, 2]; title = "training data: T")
heatmap!(ax, lon, lat, Array(batch[:,:,1 + (j-1) * M,1]); colorrange1, colormap = :thermometer)
display(fig2)
=#