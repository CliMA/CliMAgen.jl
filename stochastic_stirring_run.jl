FT = Float32
#Read in from toml
batchsize = t_sims
inchannels = 1
sigma_min = FT(1e-2);
sigma_max = FT(sigmax);
nwarmup = 5000;
gradnorm = FT(1.0);
learning_rate = FT(2e-4);
beta_1 = FT(0.9);
beta_2 = FT(0.999);
epsilon = FT(1e-8);
ema_rate = FT(0.999);
device = Flux.gpu

# Create network
net = NoiseConditionalScoreNetwork(;
                                    noised_channels = inchannels,
                                    shift_input = true,
                                    shift_output = true,
                                    mean_bypass = true,
                                    scale_mean_bypass = true,
                                    gnorm = true,
                                    )
model = VarianceExplodingSDE(sigma_max, sigma_min, net)
model = device(model)
model_smooth = deepcopy(model)
opt = Flux.Optimise.Optimiser(WarmupSchedule{FT}(nwarmup),
                              Flux.Optimise.ClipNorm(gradnorm),
                              Flux.Optimise.Adam(learning_rate,(beta_1, beta_2), epsilon)
) 
opt_smooth = ExponentialMovingAverage(ema_rate)

# model parameters
ps = Flux.params(model)
# setup smoothed parameters
ps_smooth = Flux.params(model_smooth)
lossfn = x -> score_matching_loss(model, x)

function mock_callback(batch; ps = ps, opt = opt, lossfn = lossfn, ps_smooth = ps_smooth, opt_smooth = opt_smooth)
    grad = Flux.gradient(() -> sum(lossfn(batch)), ps)
    Flux.Optimise.update!(opt, ps, grad)
    Flux.Optimise.update!(opt_smooth, ps_smooth, ps)
    return nothing
end
##
batch = zeros(FT, Ny, Ny, 1, t_sims);
gradient_steps = 200
sigmaxs = []
for j in ProgressBar(1:gradient_steps)
    for (i,simulation) in enumerate(sims)
        run!(simulation,period=Day(1),output=true)
        tmp = FT.((simulation.model.output.vor .- μ) / σ)
        rtmp = tmp[1:2:end, :] + tmp[2:2:end, :]
        batch[:, :, 1, i] .= rtmp
        rm("run_0001", recursive=true) 
    end
    sigmax =  maximum([norm(batch[:,:,1,i] - batch[:, :, 1, j]) for i in 1:t_sims, j in 1:t_sims if i != j]) 
    push!(sigmaxs , sigmax)
    mock_callback(device(batch))
end

##
nsamples = 1
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

heatmap(Array(samples[:,:,1,1]))