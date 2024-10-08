using Flux
using CUDA
using cuDNN
using Dates
using Random
using TOML

using CliMAgen
using CliMAgen: dict2nt
using CliMAgen: VarianceExplodingSDE, NoiseConditionalScoreNetwork
using CliMAgen: score_matching_loss
using CliMAgen: WarmupSchedule, ExponentialMovingAverage
using CliMAgen: train!, load_model_and_optimizer

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/conus404/preprocessing_utils.jl"))
include(joinpath(package_dir,"examples/utils_data.jl")) # for data loading
include("analysis.jl") # for analysis

function run_training(params; FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize

    standard_scaling = params.data.standard_scaling
    low_pass = params.data.low_pass
    low_pass_k = params.data.low_pass_k
    fname_train = params.data.fname_train
    fname_test = params.data.fname_test
    precip_channel = params.data.precip_channel
    precip_floor::FT = params.data.precip_floor

    # we always train with the preprocessing parameters derived from the
    # training data.
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_train.jld2")

    sigma_max::FT = params.model.sigma_max
    sigma_min::FT = params.model.sigma_min
    dropout_p::FT = params.model.dropout_p
    inchannels = params.model.inchannels
    shift_input = params.model.shift_input
    shift_output = params.model.shift_output
    mean_bypass = params.model.mean_bypass
    scale_mean_bypass = params.model.scale_mean_bypass
    gnorm = params.model.gnorm
    proj_kernelsize = params.model.proj_kernelsize
    outer_kernelsize = params.model.outer_kernelsize
    middle_kernelsize = params.model.middle_kernelsize
    inner_kernelsize = params.model.inner_kernelsize
    periodic = params.model.periodic

    nwarmup = params.optimizer.nwarmup
    gradnorm::FT = params.optimizer.gradnorm
    learning_rate::FT = params.optimizer.learning_rate
    beta_1::FT = params.optimizer.beta_1
    beta_2::FT = params.optimizer.beta_2
    epsilon::FT = params.optimizer.epsilon
    ema_rate::FT = params.optimizer.ema_rate

    nepochs = params.training.nepochs
    freq_chckpt = params.training.freq_chckpt

    # set up rng
    rngseed > 0 && Random.seed!(rngseed)

    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        @info "Training on GPU"
    else
        device = Flux.cpu
        @info "Training on CPU"
    end

    # set up dataset
    dataloaders = get_data_conus404(fname_train, fname_test, precip_channel, batchsize;
        precip_floor = precip_floor, FT=FT, preprocess_params_file=preprocess_params_file)
    
    # set up model and optimizers
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    loss_file = joinpath(savedir, "losses.txt")
    
    if isfile(checkpoint_path) && isfile(loss_file)
        BSON.@load checkpoint_path model model_smooth opt opt_smooth
        model = device(model)
        model_smooth = device(model_smooth)
        loss_data = DelimitedFiles.readdlm(loss_file, ',', skipstart = 1)
        start_epoch = loss_data[end,1]+1
    else
        net = NoiseConditionalScoreNetwork(;
                                           noised_channels = inchannels,
                                           dropout_p = dropout_p,
                                           shift_input = shift_input,
                                           shift_output = shift_output,
                                           mean_bypass = mean_bypass,
                                           scale_mean_bypass = scale_mean_bypass,
                                           gnorm = gnorm,
                                           proj_kernelsize = proj_kernelsize,
                                           outer_kernelsize = outer_kernelsize,
                                           middle_kernelsize = middle_kernelsize,
                                           inner_kernelsize = inner_kernelsize,
                                           periodic = periodic,
                                           )
        model = VarianceExplodingSDE(sigma_max, sigma_min, net)
        model = device(model)
        model_smooth = deepcopy(model)
    
        opt = Flux.Optimise.Optimiser(
            WarmupSchedule{FT}(
                nwarmup 
            ),
            Flux.Optimise.ClipNorm(gradnorm),
            Flux.Optimise.Adam(
                learning_rate,
                (beta_1, beta_2),
                epsilon
            )
        )
        opt_smooth = ExponentialMovingAverage(ema_rate)

        # set up loss file
        loss_names = reshape(["#Epoch", "Mean Train", "Spatial Train","Mean Test","Spatial Test"], (1,5))
        open(loss_file,"w") do io
             DelimitedFiles.writedlm(io, loss_names,',')
        end

        start_epoch=1
    end

    # set up loss function
    lossfn = x -> score_matching_loss(model, x)

    # train the model
    train!(
        model,
        model_smooth,
        lossfn,
        dataloaders,
        opt,
        opt_smooth,
        nepochs,
        device;
        start_epoch = start_epoch,
        savedir = savedir,
        logger = nothing,
        freq_chckpt = freq_chckpt,
    )
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    # set up directory for saving checkpoints
    !ispath(params.experiment.savedir) && mkpath(params.experiment.savedir)

    run_training(params; FT=FT)

    if :sampling in keys(params)
        run_analysis(params; FT=FT)
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml = ARGS[1])
end
