using SpeedyWeather
using StochasticStir
using GLMakie
using LinearAlgebra
using Statistics
using ProgressBars
using Flux
using CliMAgen

# model components
sims = []
t_sims = 32
for in in 1:t_sims
    spectral_grid = SpectralGrid(trunc=42,nlev=1)   
    forcing = StochasticStirring(spectral_grid,latitude=45,strength=7e-11)
    drag = JetDrag(spectral_grid,time_scale=Day(6))
    initial_conditions = StartFromRest()
    # construct the model and initialize
    model = BarotropicModel(;spectral_grid,initial_conditions,forcing,drag)
    push!(sims,initialize!(model) )
end

Nx, Ny = size(sims[1].model.output.vor)
# now run and store output
batch = zeros(Nx, Ny, 1, t_sims);

spin_up_days = 100
for (i,simulation) in enumerate(sims)
    run!(simulation,period=Day(spin_up_days),output=true)
    batch[:, :, 1, i] .= simulation.model.output.vor
    rm("run_0001", recursive=true) 
end
μ = mean(batch[:,:,1,:])
σ = std(batch[:,:,1,:])
sigmax =  maximum([norm(batch[:,:,1,i] - batch[:, :, 1, j]) for i in 1:t_sims, j in 1:t_sims if i != j]) / σ

##
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
device = Flux.cpu

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
batch = zeros(FT, Nx, Ny, 1, t_sims);
gradient_steps = 100
for j in ProgressBar(1:gradient_steps)
    for (i,simulation) in enumerate(sims)
        run!(simulation,period=Day(1),output=true)
        batch[:, :, 1, i] .= FT.((simulation.model.output.vor .- μ) / σ)
        rm("run_0001", recursive=true) 
    end
    mock_callback(device(batch))
end

