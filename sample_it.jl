using CliMAgen
using BSON 
using GLMakie
using LinearAlgebra
using Statistics
using ProgressBars
using Flux

device = Flux.gpu

# μ = 2.6706336894504953e-7 
# σ = 1.6521036461155343e-5
# to compare to speedy weather outout need to rescale like y = x * σ + μ
# where x is the sampled image and y is the speedy weather output
# alternatively can rescale speedy weather like x = (y - μ) / σ
@info "Loading Model"
checkpoint_path = "checkpoint.bson"
BSON.@load checkpoint_path model model_smooth opt opt_smooth
model_smooth = device(model_smooth)


@info "Setting up Sampler"
nsamples = 9
nsteps = 250
resolution = 64
inchannels = 1
time_steps, Δt, init_x = setup_sampler(
    model_smooth,
    device,
    resolution,
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