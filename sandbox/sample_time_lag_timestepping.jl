


nsteps = 10
us = zeros(Float32, 64, 64, nsteps);
ctrain[:, :, [1], [1]] .= randn(Float32, 64, 64, 1, 1)
us[:, :, 1] .= ctrain[:, :, 1, 1]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = GLMakie.Axis(fig[ii,jj]; title = "Day $(i+80)")
    GLMakie.heatmap!(ax, us[:,:,i+80], colorrange = (-1.5, 1.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag1.png", fig)

##

nsteps = 10
us = zeros(Float32, 64, 64, nsteps);
ctrain[:, :, [1], [1]] .= randn(Float32, 64, 64, 1, 1)
us[:, :, 1] .= ctrain[:, :, 1, 1]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = GLMakie.Axis(fig[ii,jj]; title = "Day $(i-1)")
    GLMakie.heatmap!(ax, us[:,:,i], colorrange = (-1.5, 1.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag1_noise_start.png", fig)

##

nsteps = 10
us = zeros(Float32, 64, 64, nsteps);
ctrain[:, :, [1], [1]] .= zeros(Float32, 64, 64, 1, 1)
us[:, :, 1] .= ctrain[:, :, 1, 1]
i=2
for i in ProgressBar(2:nsteps)
    samples = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:9 
    ii = (i-1)%3 + 1 
    jj = (i-1)÷3 + 1
    ax = GLMakie.Axis(fig[ii,jj]; title = "Day $(i-1)")
    GLMakie.heatmap!(ax, us[:,:,i], colorrange = (-1.5, 1.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag1_zero_start.png", fig)

##
init_x .= x0
rng = MersenneTwister(1234*100)
ctrain[:, :, :, :] .= cpu_batch[:, :, 2:3, :]
nsteps = 9
us = zeros(Float32, 64, 64, nsteps);
ctrain[:, :, [1], [1]] .+= randn(Float32, 64, 64, 1, 1) * 6 # 6 is okay
us[:, :, 1] .= ctrain[:, :, 1, 1] 
for i in ProgressBar(2:nsteps)
    # init_x .= x0 # to check for stability if it gets modified too much from iteration to iteration
    samples = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:minimum([nsteps, 9])
    jj = (i-1)%3 + 1 
    ii = (i-1)÷3 + 1
    ax = GLMakie.Axis(fig[ii,jj]; title = "Day $(i-1)")
    GLMakie.heatmap!(ax, us[:,:,i+11], colorrange = (-1.5, 1.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag1_big_noise_start.png", fig)

##
init_x .= x0
rng = MersenneTwister(1234*100)
ctrain[:, :, :, :] .= cpu_batch[:, :, 2:3, :]
nsteps = 40
us = zeros(Float32, 64, 64, nsteps);
ctrain[:, :, [1], [1]] .= ones(Float32, 64, 64, 1, 1) * 10 # 6 is okay
us[:, :, 1] .= ctrain[:, :, 1, 1] 
for i in ProgressBar(2:nsteps)
    # init_x .= x0 # to check for stability if it gets modified too much from iteration to iteration
    samples = Euler_Maruyama_sampler(score_model, init_x, time_steps, Δt; c=ctrain[:, :, :, 1:nsamples], rng = rng)
    samples = cpu(samples)
    us[:, :, i] .= samples[:, :, 1, 1]
    ctrain[:, :, [1], [1]] .= samples[:, :, [1], [1]]
end

fig = Figure() 
for i in 1:minimum([nsteps, 9])
    jj = (i-1)%3 + 1 
    ii = (i-1)÷3 + 1
    ax = GLMakie.Axis(fig[ii,jj]; title = "Day $(i-1)")
    GLMakie.heatmap!(ax, us[:,:,i + 31], colorrange = (-1.5, 1.5), colormap = :balance, interpolate = true)
end
save("timestepping_lag1_ones_start.png", fig)