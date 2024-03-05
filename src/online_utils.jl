
function mock_callback(batch; ps = ps, opt = opt, lossfn = lossfn, ps_smooth = ps_smooth, opt_smooth = opt_smooth)
    grad = Flux.gradient(() -> sum(lossfn(batch)), ps)
    Flux.Optimise.update!(opt, ps, grad)
    Flux.Optimise.update!(opt_smooth, ps_smooth, ps)
    return nothing
end

function saving_callback(path; model = model, opt = opt, model_smooth = model_smooth, opt_smooth=opt_smooth)
    save_model_and_optimizer(Flux.cpu(model), Flux.cpu(model_smooth), opt, opt_smooth, path)
    @info "Checkpoint saved to $(path)."
end


function setup_model(params; FT=Float32)
    batchsize = params.data.batchsize
    inchannels = params.model.inchannels
    sigma_min::FT = params.model.sigma_min
    sigma_max::FT = params.model.sigma_max
    inchannels = params.model.inchannels
    shift_input = params.model.shift_input
    shift_output = params.model.shift_output
    mean_bypass = params.model.mean_bypass
    scale_mean_bypass = params.model.scale_mean_bypass
    gnorm = params.model.gnorm
    nwarmup = params.optimizer.nwarmup
    gradnorm::FT = params.optimizer.gradnorm
    learning_rate::FT = params.optimizer.learning_rate
    beta_1::FT = params.optimizer.beta_1
    beta_2::FT = params.optimizer.beta_2
    epsilon::FT = params.optimizer.epsilon
    ema_rate::FT = params.optimizer.ema_rate
    
    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Training on GPU"
    else
        device = Flux.cpu
        @info "Training on CPU"
    end
    # Create network
    net = NoiseConditionalScoreNetwork(;
                                        noised_channels = inchannels,
                                        shift_input = shift_input,
                                        shift_output = shift_output,
                                        mean_bypass = mean_bypass,
                                        scale_mean_bypass = scale_mean_bypass,
                                        gnorm = gnorm,
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
    return model, model_smooth, opt, opt_smooth, ps, ps_smooth, lossfn
end