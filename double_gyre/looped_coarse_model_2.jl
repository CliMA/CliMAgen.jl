using CliMAgen, Flux, HDF5, Random, ProgressBars, Statistics, BSON, LinearAlgebra
using CairoMakie
const gfp_scale = 1
Random.seed!(1234)
const extra_scale = 2

# train differently, t = 0 and t = 1 
# condition using different information (such as global and ensemble average mean surface)
FT = Float32
data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"
save_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/DoubleGyre/"
include("sampler.jl")

factors = [2^k for k in 1:7]
level_indices = 8:8
for level_index in ProgressBar(level_indices )
M = 128
casevar = 5
factor = 2
hfile = h5open(data_directory_training * "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar)_complement.hdf5", "r")
field = FT.(read(hfile["field $factor"]))
close(hfile)

is = rand(1:size(field, 4),100)
js = rand(1:size(field, 4),100)
nlast = size(field, 3) - 1
sigma_max = maximum([norm(field[:, :, 1:nlast, i] - field[:, :, 1:nlast, j]) for i in is, j in js])

# ADAM parameters
nwarmup = 5000
gradnorm = FT(1.0);
learning_rate = FT(2e-4);
beta_1 = FT(0.9);
beta_2 = FT(0.999);
epsilon = FT(1e-8);
ema_rate = FT(0.999);
# Optimization
device = Flux.gpu
inchannels = size(field, 3) - 1
context_channels = 1
sigma_min = FT.(1e-2)
sigma_max = FT.(sigma_max)

@info "defining the network"
# Define Network
quick_arg = true
kernel_size = 3
kernel_sizes =  [3, 2, 1, 0] #  [0, 0, 0, 0] #   
channel_scale = 2
net = NoiseConditionalScoreNetwork(;
                                    channels = channel_scale .* [32, 64, 128, 256],
                                    proj_kernelsize   = kernel_size + kernel_sizes[1],
                                    outer_kernelsize  = kernel_size + kernel_sizes[2],
                                    middle_kernelsize = kernel_size + kernel_sizes[3],
                                    inner_kernelsize  = kernel_size + kernel_sizes[4],
                                    noised_channels = inchannels,
                                    context_channels = context_channels,
                                    context = true,
                                    shift_input = quick_arg,
                                    shift_output = quick_arg,
                                    mean_bypass = quick_arg,
                                    scale_mean_bypass = quick_arg,
                                    gnorm = quick_arg,
                                    )
score_model = VarianceExplodingSDE(sigma_max, sigma_min, net)
score_model = device(score_model)
score_model_smooth = deepcopy(score_model)
opt = Flux.Optimise.Optimiser(WarmupSchedule{FT}(nwarmup),
                            Flux.Optimise.ClipNorm(gradnorm),
                            Flux.Optimise.Adam(learning_rate,(beta_1, beta_2), epsilon)
) 
opt_smooth = ExponentialMovingAverage(ema_rate);
# model parameters
ps = Flux.params(score_model);
# setup smoothed parameters
ps_smooth = Flux.params(score_model_smooth);

function lossfn_c(y; noised_channels = inchannels, context_channels=context_channels)
    x = y[:,:,1:noised_channels,:]
    c = y[:,:,(noised_channels+1):(noised_channels+context_channels),:]
    return vanilla_score_matching_loss(score_model, x; c)
end
function mock_callback(batch; ps = ps, opt = opt, lossfn = lossfn_c, ps_smooth = ps_smooth, opt_smooth = opt_smooth)
    grad = Flux.gradient(() -> sum(lossfn(batch)), ps)
    Flux.Optimise.update!(opt, ps, grad)
    Flux.Optimise.update!(opt_smooth, ps_smooth, ps)
    return nothing
end


for factor in ProgressBar(factors)
    FT = Float32
    M = 128 
    casevar = 5
    prefix = "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar)_complement_$(factor)_"
    figure_directory = "DoubleGyreFigures/"

    hfile = h5open(data_directory_training * "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar)_complement.hdf5", "r")
    field = read(hfile["field $factor"])
    close(hfile)

    field = FT.(field[:, :, :, :])

    ##
    @info "getting read to train"
    FT = Float32
    batchsize = 32
    Ma, Mb, Mc, M = size(field)
    field = FT.(field[:, :, :, 1:M]) # no shuffle on the field to start
    Ntest = M ÷ 10
    N = M - Ntest ≥ 0 ? M - Ntest : 1
    skipind = N ÷ batchsize
    collections = [i:skipind:N for i in 1:skipind-1]
    skipind2 = Ntest ÷ batchsize
    collections_test = [N+i:skipind2:N + Ntest for i in 1:skipind2-1]

    if factor == factors[1]
        epochs = 1000
    else 
        epochs = 201 
    end

    contextfield = zeros(FT, Ma, Mb, 1, 2)
    contextind1 = M 
    contextind2 = N + (M - N) ÷ 2
    contextfield[:, :, 1, 1] .= field[:, :, end:end, contextind1]
    contextfield[:, :, 1, 2] .= field[:, :, end:end, contextind2]

    losses = []
    losses_test = []
    for epoch in ProgressBar(1:epochs)
        shuffled_indices = shuffle(1:N)
        for collection in collections
            shuffled_collection = shuffled_indices[collection]
            batch = field[:,:,:, shuffled_collection]
            mock_callback(device(batch))
        end
        # evaluate loss 
        if epoch % 1 == 0
            lossvalue = Float32.([0.0])
            for collection in collections
                y = field[:,:,:,collection]
                lossvalue .+= lossfn_c(device(y)) / length(collections)
            end
            push!(losses, lossvalue)
            lossvalue = Float32.([0.0])
            for collection in collections_test
                y = field[:,:,:,collection]
                lossvalue .+= lossfn_c(device(y)) / length(collections_test)
            end
            push!(losses_test, lossvalue)
        end
        if epoch % 100 == 0
            @info "saving model"
            CliMAgen.save_model_and_optimizer(Flux.cpu(score_model), Flux.cpu(score_model_smooth), opt, opt_smooth, save_directory  * "double_gyre_factor_$(factor)_$(epoch)_complement.bson")
        end
    end

    hfile = h5open(save_directory * "double_gyre_losses_$(factor)_$(casevar)_complement.hdf5", "w")
    hfile["losses"] = [loss[1] for loss in losses]
    hfile["losses_test"] = [loss[1] for loss in losses_test]
    close(hfile)


    # /orcd/data/raffaele/001/sandre/OceananigansData
    nsamples = 100
    nsteps = 300 
    resolution = size(field)[1:2]
    time_steps, Δt, init_x = setup_sampler(
        score_model_smooth,
        device,
        resolution,
        inchannels;
        num_images=nsamples,
        num_steps=nsteps,
    )


    ntotal = nsamples * 2
    total_samples = zeros(resolution..., inchannels, ntotal)
    cprelim = zeros(resolution..., context_channels, nsamples)
    rng = MersenneTwister(12345)
    tot = ntotal ÷ nsamples
    for i in ProgressBar(1:tot)
        if i ≤ (tot ÷ 2)
            # cprelim = reshape(gfp(Float32(0.1)), (192, 96, 1, 1)) * gfp_scale
            cprelim .= contextfield[:, :, :, 1:1] # ensemble_mean[:, :, :, 1:1]
        else
            # cprelim = reshape(gfp(Float32(1.0)), (192, 96, 1, 1)) * gfp_scale
            cprelim .= contextfield[:, :, :, end:end]# ensemble_mean[:, :, :, end:end]
        end
        c = device(cprelim)
        samples = Array(Euler_Maruyama_sampler(score_model_smooth, init_x, time_steps, Δt; rng, c))
        total_samples[:, :, :, (i-1)*nsamples+1:i*nsamples] .= samples
    end

    averaged_samples_1 = mean(total_samples[:,:, :, 1:nsamples], dims = 4)[:, :, :, 1]
    std_samples_1 = std(total_samples[:,:, :, 1:nsamples], dims = 4)[:, :, :, 1]
    averaged_samples_2 = mean(total_samples[:,:, :, nsamples+1:2 * nsamples], dims = 4)[:, :, :, 1]
    std_samples_2 = std(total_samples[:,:, :, nsamples+1:2 * nsamples], dims = 4)[:, :, :, 1]


    hfile = h5open(save_directory * prefix * "generative_samples.hdf5", "w" )
    hfile["samples context 1"] = total_samples[:, :, :, 1:nsamples]
    hfile["samples context 2"] = total_samples[:, :, :, nsamples+1:2 * nsamples]
    hfile["context field 1"] = contextfield[:, :, :, 1:1]
    hfile["context field 2"] = contextfield[:, :, :, end:end]
    hfile["averaged samples 1"] = averaged_samples_1
    hfile["std samples 1"] = std_samples_1
    hfile["averaged samples 2"] = averaged_samples_2
    hfile["std samples 2"] = std_samples_2
    hfile["last training index"] = N
    hfile["sample index 1"] = contextind1
    hfile["sample index 2"] = contextind2
    close(hfile)

    fig = Figure()
    ax = Axis(fig[1,1]; title = "losses")
    lines!(ax, [loss[1] for loss in losses], color = :blue)
    lines!(ax, [loss[1] for loss in losses_test], color = :red)
    save(figure_directory  * "losses_double_gyre_$(casevar)_complement_$(factor)_$(level_index)_complement.png", fig)


    fig = Figure(resolution = (2400, 800))
    stateindex = 1
    ηmax = maximum(contextfield[:, :, 1, 1])
    ηrange = (-ηmax, ηmax)
    colormap_η = :balance
    colormap = :balance
    ax = Axis(fig[1, 1]; title = "context")
    heatmap!(ax, contextfield[:, :, 1, 1]; colormap = colormap_η, colorrange = ηrange  )
    quval = quantile(field[:, :, stateindex, contextind1][:], 0.95)
    crange = (-quval, quval)
    ax = Axis(fig[1, 2]; title = "ground truth")
    heatmap!(ax, field[:, :, stateindex, contextind1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 3]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 4]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 5]; title = "mean")
    heatmap!(ax, averaged_samples_1[:, :, stateindex], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 6]; title = "std")
    heatmap!(ax, std_samples_1[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

    ax = Axis(fig[2, 1]; title = "context")
    heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
    ax = Axis(fig[2, 2]; title = "ground truth")
    heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 3]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 4]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 5]; title = "mean")
    heatmap!(ax, averaged_samples_2[:, :, stateindex], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 6]; title = "std")
    heatmap!(ax, std_samples_2[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

    save(figure_directory  * "double_gyre_samples_u_case_$(factor)_$(level_index)_complement.png", fig)

    fig = Figure(resolution = (2400, 800))
    stateindex = 2
    colormap = :balance
    ax = Axis(fig[1, 1]; title = "context")
    heatmap!(ax, contextfield[:, :, 1, 1]; colormap = colormap_η, colorrange = ηrange  )
    quval = quantile(field[:, :, stateindex, contextind1][:], 0.95)
    crange = (-quval, quval)
    ax = Axis(fig[1, 2]; title = "ground truth")
    heatmap!(ax, field[:, :, stateindex, contextind1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 3]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 4]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 5]; title = "mean")
    heatmap!(ax, averaged_samples_1[:, :, stateindex], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 6]; title = "std")
    heatmap!(ax, std_samples_1[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

    ax = Axis(fig[2, 1]; title = "context")
    heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
    ax = Axis(fig[2, 2]; title = "ground truth")
    heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 3]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 4]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 5]; title = "mean")
    heatmap!(ax, averaged_samples_2[:, :, stateindex], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 6]; title = "std")
    heatmap!(ax, std_samples_2[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

    save(figure_directory  * "double_gyre_samples_v_$(factor)_$(level_index)_complement.png", fig)

    fig = Figure(resolution=(2400, 800))
    stateindex = 3
    colormap = :balance
    ax = Axis(fig[1, 1]; title="context")
    heatmap!(ax, contextfield[:, :, 1, 1]; colormap=colormap_η, colorrange=ηrange)
    quval = quantile(field[:, :, stateindex, contextind1][:], 0.95)
    crange = (-quval, quval)
    ax = Axis(fig[1, 2]; title="ground truth")
    heatmap!(ax, field[:, :, stateindex, contextind1], colormap=colormap, colorrange=crange)
    ax = Axis(fig[1, 3]; title="samples")
    heatmap!(ax, total_samples[:, :, stateindex, 1], colormap=colormap, colorrange=crange)
    ax = Axis(fig[1, 4]; title="samples")
    heatmap!(ax, total_samples[:, :, stateindex, 2], colormap=colormap, colorrange=crange)
    ax = Axis(fig[1, 5]; title = "mean")
    heatmap!(ax, averaged_samples_1[:, :, stateindex], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 6]; title = "std")
    heatmap!(ax, std_samples_1[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))


    ax = Axis(fig[2, 1]; title="context")
    heatmap!(ax, contextfield[:, :, 1, end]; colormap=colormap_η, colorrange=ηrange)
    ax = Axis(fig[2, 2]; title="ground truth")
    heatmap!(ax, field[:, :, stateindex, contextind2], colormap=colormap, colorrange=crange)
    ax = Axis(fig[2, 3]; title="samples")
    heatmap!(ax, total_samples[:, :, stateindex, end], colormap=colormap, colorrange=crange)
    ax = Axis(fig[2, 4]; title="samples")
    heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap=colormap, colorrange=crange)
    ax = Axis(fig[2, 5]; title = "mean")
    heatmap!(ax, averaged_samples_2[:, :, stateindex], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 6]; title = "std")
    heatmap!(ax, std_samples_2[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

    save(figure_directory  * "double_gyre_samples_w_$(factor)_$(level_index)_complement.png", fig)

    fig = Figure(resolution = (2400, 800))
    stateindex = 4
    colormap = :thermometer
    ax = Axis(fig[1, 1]; title = "context")
    heatmap!(ax, contextfield[:, :, 1, 1]; colormap = colormap_η, colorrange = ηrange  )
    quval = quantile(field[:, :, stateindex, contextind1][:], [0.05, 0.95])
    crange = quval
    ax = Axis(fig[1, 2]; title = "ground truth")
    heatmap!(ax, field[:, :, stateindex, contextind1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 3]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, 1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 4]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, 2], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 5]; title = "mean")
    heatmap!(ax, averaged_samples_1[:, :, stateindex], colormap = colormap, colorrange = crange)
    ax = Axis(fig[1, 6]; title = "std")
    heatmap!(ax, std_samples_1[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

    ax = Axis(fig[2, 1]; title = "context")
    heatmap!(ax, contextfield[:, :, 1, end]; colormap = colormap_η, colorrange = ηrange  )
    ax = Axis(fig[2, 2]; title = "ground truth")
    heatmap!(ax, field[:, :, stateindex, contextind2], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 3]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, end], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 4]; title = "samples")
    heatmap!(ax, total_samples[:, :, stateindex, end-1], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 5]; title = "mean")
    heatmap!(ax, averaged_samples_2[:, :, stateindex], colormap = colormap, colorrange = crange)
    ax = Axis(fig[2, 6]; title = "std")
    heatmap!(ax, std_samples_2[:, :, stateindex], colormap = :viridis, colorrange = (0, quantile(std_samples_1[:, :, stateindex][:], 0.95)))

    save(figure_directory  * "double_gyre_samples_b_$(factor)_$(level_index)_complement.png", fig)



    stdfield = std(field[:, :, :, 1:N], dims = 4)
    meanfield = mean(field[:, :, :, 1:N], dims = 4)

    fig = Figure(resolution = (2400, 800))
    for i in 1:5
        crange = (-quantile(meanfield[:, :, i, 1][:], 0.95), quantile(meanfield[:, :, i, 1][:], 0.95))
        ax = Axis(fig[1, i]; title = "mean $i")
        heatmap!(ax, meanfield[:, :, i, 1], colormap = :balance, colorrange = crange)
        ax = Axis(fig[2, i]; title = "std $i")
        heatmap!(ax, stdfield[:, :, i, 1], colormap = :viridis, colorrange = (0, quantile(stdfield[:, :, i, 1][:], 0.95)))
    end

    save(figure_directory  * "double_gyre_mean_std_$(factor)_$(level_index)_complement.png", fig)

end
end