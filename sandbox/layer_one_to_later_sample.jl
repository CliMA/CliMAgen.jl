using GLMakie, Random
include("sampler.jl")


nsamples = 8
nsteps = 250
resolution = (128, 64)

rng = MersenneTwister(1234)

ctrain = zeros(Float32, resolution..., length(my_fields), nsamples);

lat, lon = RingGrids.get_latdlonds(my_fields[1].var);
lon = reshape(lon, (128, 64))[:,1];
lat = reshape(lat, (128, 64))[1, :];
rbatch = copy(reshape(gated_array, (128, 64, 2*length(my_fields), batchsize)));
batch = (rbatch .- reshape(μ, (1, 1, length(my_fields) * 2, 1))) ./ reshape(σ, (1, 1, length(my_fields) * 2, 1));

ctrain[:, :, :, :] .= Float32.(batch[:, :, (1 + length(my_fields)):end, [8]]);

time_steps, Δt, init_x = setup_sampler(
    score_model_smooth,
    device,
    resolution,
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)

x0 = copy(init_x);

samples = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng);
samples = cpu(samples);


for ind in eachindex(my_fields)
    p = 0.03
    colorrange = (quantile(batch[:, :, ind+length(my_fields), 8][:], p), quantile(batch[:, :, ind+length(my_fields), 8][:], 1 - p))
    colorrange_difference = (-0.5, 0.5)
    fig = Figure()
    for i in 1:2
        ax = Axis(fig[i, 1]; title = "ai current timestep sample $i")
        heatmap!(ax,lon, lat, ctrain[:,:,ind,i]; colorrange, colormap = :thermometer, interpolate = true)
        ax = Axis(fig[i, 2]; title = "ai next timestep sample $i")
        heatmap!(ax,lon, lat, samples[:,:,ind,i]; colorrange, colormap = :thermometer, interpolate = true)
        ax = Axis(fig[i, 3]; title = "ai difference $i")
        heatmap!(ax,lon, lat, samples[:,:,ind,i] - ctrain[:,:,ind,i]; colorrange = colorrange_difference, colormap = :balance, interpolate = true)
    end
    ax = Axis(fig[3, 1]; title = "training data current timestep")
    heatmap!(ax,lon, lat, batch[:,:,ind + length(my_fields),8]; colorrange, colormap = :thermometer, interpolate = true)
    ax = Axis(fig[3, 2]; title = "training data next timestep")
    heatmap!(ax,lon, lat, batch[:,:,ind,8]; colorrange, colormap = :thermometer, interpolate = true)
    ax = Axis(fig[3, 3]; title = "training difference")
    heatmap!(ax,lon, lat, batch[:,:,ind,8] - batch[:,:,ind + length(my_fields),8]; colorrange = colorrange_difference, colormap = :balance, interpolate = true)
    display(fig)
    save("layers_to_later_$ind.png", fig)
end

#=
p = 0.03

M = length(fields)
for ind in 1:4
    colorrange = (quantile(samples[:, :, ind, :][:], p), quantile(samples[:, :, ind, :][:], 1-p))
    colorrange_difference = (-0.5, 0.5)
    fig = Figure()
    i = 1
    for i in 1:2
        ax = Axis(fig[i, 1]; title = "ai current timestep sample $i")
        heatmap!(ax,lon, lat, ctrain[:,:,ind,i]; colorrange, colormap = :thermometer, interpolate = true)
        ax = Axis(fig[i, 2]; title = "ai next timestep sample $i")
        heatmap!(ax,lon, lat, samples[:,:,ind,i]; colorrange, colormap = :thermometer, interpolate = true)
        ax = Axis(fig[i, 3]; title = "ai difference $i")
        heatmap!(ax,lon, lat, samples[:,:,ind,i] - ctrain[:,:,ind,i]; colorrange = colorrange_difference, colormap = :balance, interpolate = true)
    end
    ax = Axis(fig[3, 1]; title = "training data current timestep")
    heatmap!(ax,lon, lat, batch[:,:,ind + M,8]; colorrange, colormap = :thermometer, interpolate = true)
    ax = Axis(fig[3, 2]; title = "training data next timestep")
    heatmap!(ax,lon, lat, batch[:,:,ind,8]; colorrange, colormap = :thermometer, interpolate = true)
    ax = Axis(fig[3, 3]; title = "training difference $i")
    heatmap!(ax,lon, lat, batch[:,:,ind,8] - batch[:,:, ind + M,8]; colorrange = colorrange_difference, colormap = :balance, interpolate = true)
    display(fig)
    save("layer_one_to_later_$ind.png", fig)
end

@info "Timestepping"
ctrain[:, :, 1, :] .= Float32.(batch[:, :, 2, [8]])
dynamic_steps = 365
us = zeros(Float32, 128, 64, nsamples, dynamic_steps);
us[:, :, :, 1] .= ctrain[:, :, 1, 1:nsamples]
for i in ProgressBar(2:dynamic_steps)
        time_steps, Δt, init_x = setup_sampler(
        score_model_smooth,
        device,
        resolution,
        inchannels;
        num_images=nsamples,
        num_steps=nsteps,
    )
    samples = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, :, i] .= ctrain[:, :, 1, 1:nsamples]
    ctrain .= samples
end

fig2 = Figure() 
ax = Axis(fig2[1,1]; title = "timestep $dynamic_steps")
heatmap!(ax,lon, lat, us[:,:,1,end]; colorrange, colormap = :thermometer, interpolate = true)
display(fig2)
=#


