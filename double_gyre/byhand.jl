using CliMAgen, Flux, HDF5, Random, ProgressBars, Statistics, BSON
const gfp_scale = 1
Random.seed!(1234)
const extra_scale = 2

# train differently, t = 0 and t = 1 
# condition using different information (such as global and ensemble average mean surface)
FT = Float32
include("utils.jl")
include("process_data.jl")
field = FT.(field[:, :, :, :])

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

# Define Network
quick_arg = true
kernel_size = 3
kernel_sizes =   [3, 2, 1, 0] #  [0, 0, 0, 0] #   
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

##
batchsize = 64
Ma, Mb, Mc, M = size(field)
field = field[:, :, :, 1:M] # no shuffle on the field to start
Ntest = M ÷ 10
N = M - Ntest ≥ 0 ? M - Ntest : 1
skipind = N ÷ batchsize
collections = [i:skipind:N for i in 1:skipind-1]
skipind2 = Ntest ÷ batchsize
collections_test = [N+1:N + Ntest]
epochs = 500

contextfield = zeros(Ma, Mb, 1, 2)
contextfield[:, :, 1, 1] .= field[:, :, end:end, N+1]
contextfield[:, :, 1, 2] .= field[:, :, end:end, N+2]

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
        CliMAgen.save_model_and_optimizer(Flux.cpu(score_model), Flux.cpu(score_model_smooth), opt, opt_smooth, "double_gyre_$epoch.bson")
    end
end

hfile = h5open("double_gyre_losses.hdf5", "w")
hfile["losses"] = [loss[1] for loss in losses]
hfile["losses_test"] = [loss[1] for loss in losses_test]
close(hfile)