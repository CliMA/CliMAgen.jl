using CliMAgen
using Flux
using ProgressBars
using LinearAlgebra
using UnicodePlots
FT = Float32
#Read in from toml
batchsize = 16
inchannels = 1
n_pixels = 8
sigma_min = FT(1e-2);
sigma_max = FT(10.0);
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
function score_analytic(x, t, m) # dx = g(t) dW -> σ_t^2 =  ∫ g(t')^2 dt'
    σ_t = @. m.σ_min * (m.σ_max/m.σ_min)^t
    return @. - x/(1 + σ_t^2)    
end

sim_steps = 1000
normlist = Float32[]


function mock_callback(batch; ps = ps, opt = opt, lossfn = lossfn, ps_smooth = ps_smooth, opt_smooth = opt_smooth)
    grad = Flux.gradient(() -> sum(lossfn(batch)), ps)
    Flux.Optimise.update!(opt, ps, grad)
    Flux.Optimise.update!(opt_smooth, ps_smooth, ps)
    return nothing
end

for i in ProgressBar(1:sim_steps)
    # simulation output  
    batch = randn(FT, n_pixels,n_pixels,1, batchsize)
    mock_callback(device(batch))
    # the below is only for this example
    x_0 = batch
    sa = zeros(eltype(x_0),size(x_0))
    sm = zeros(eltype(x_0),size(x_0))
    ts = FT.(range(0, 1, length=10))
    Nt = length(ts)
    for t in ts
        z = randn(eltype(x_0),size(x_0))
        μ_t, σ_t = marginal_prob(model, x_0, t)
        x_t = @. μ_t + σ_t * z 
        sa += score_analytic(x_t, t, model)
        sm += score(model, x_t, t)
    end
    push!(normlist, norm(sa - sm))
end