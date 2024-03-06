batch = zeros(FT, Ny, Ny, 1, t_sims);
gradient_steps = 200 * 100
sigmaxs = []
for j in ProgressBar(1:gradient_steps)
    for (i,simulation) in enumerate(sims)
        run!(simulation,period=Day(1),output=true);
        tmp = FT.((simulation.model.output.vor .- μ) / σ)
        rtmp = tmp[1:2:end, :] + tmp[2:2:end, :]
        batch[:, :, 1, i] .= rtmp
        rm("run_0001", recursive=true) 
    end
    if j % 100 == 0
        sigmax =  maximum([norm(batch[:,:,1,i] - batch[:, :, 1, j]) for i in 1:t_sims, j in 1:t_sims if i != j]) 
        push!(sigmaxs , sigmax)
    end
    mock_callback(device(batch))
end
##
nsamples = 9
nsteps = 250
resolution = (Ny, Ny)
time_steps, Δt, init_x = setup_sampler(
    model_smooth,
    device,
    resolution[1],
    inchannels;
    num_images=nsamples,
    num_steps=nsteps,
)

samples = Euler_Maruyama_sampler(model_smooth, init_x, time_steps, Δt)

fig = Figure() 
for i in 1:nsamples
    ii = (i-1) ÷ 3 + 1
    jj = (i-1) % 3 + 1
    ax = Axis(fig[ii,jj])
    heatmap!(ax, Array(samples[:,:,1,i]))
end
display(fig)

#=
fig = Figure() 
for i in 1:nsamples
    ii = (i-1) ÷ 3 + 1
    jj = (i-1) % 3 + 1
    ax = Axis(fig[ii,jj])
    heatmap!(ax, Array(batch[:,:,1,i]))
end
display(fig)
=#