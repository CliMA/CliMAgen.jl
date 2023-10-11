include("GetData.jl")

using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using HDF5
using ProgressBars
using Main.GetData: get_data

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function generate_score(experiment_toml, alpha,beta,gamma,sigma;res=1)
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction = params.data.fraction
    preprocess = params.data.preprocess

    sigma_min::FT = params.model.sigma_min
    sigma_max::FT = params.model.sigma_max
    inchannels = params.model.noised_channels
    shift_input = params.model.shift_input
    shift_output = params.model.shift_output
    mean_bypass = params.model.mean_bypass
    scale_mean_bypass = params.model.scale_mean_bypass
    gnorm = params.model.gnorm
    proj_kernelsize = params.model.proj_kernelsize
    outer_kernelsize = params.model.outer_kernelsize
    middle_kernelsize = params.model.middle_kernelsize
    inner_kernelsize = params.model.inner_kernelsize
    dropout_p::FT = params.model.dropout_p
    periodic = params.model.periodic
    savedir_base = params.experiment.savedir_base
    savedir = string(savedir_base, "_preprocess_$(preprocess)_periodic_$(periodic)_$(alpha)_$(beta)_$(gamma)_$(sigma)")
    !ispath(savedir) && mkpath(savedir)
    preprocess_params_file = joinpath(savedir, "preprocess.jld2")

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
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        @info "Sampling on CPU"
    end
    f_path = "data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
    f_variable = "timeseries"
    # set up dataset
    dataloaders = get_data(
        f_path, f_variable,batchsize;
        f = fraction,
        FT=Float32,
        rng=Random.GLOBAL_RNG,
        res = res,
        preprocess = preprocess,
        preprocess_save = true,
        preprocess_params_file = preprocess_params_file,
    )

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
                                       shift_input = shift_input,
                                       shift_output = shift_output,
                                       mean_bypass = mean_bypass,
                                       scale_mean_bypass = scale_mean_bypass,
                                       gnorm = gnorm,
                                       dropout_p = dropout_p,
                                       proj_kernelsize = proj_kernelsize,
                                       outer_kernelsize = outer_kernelsize,
                                       middle_kernelsize = middle_kernelsize,
                                       inner_kernelsize = inner_kernelsize,
                                       periodic =periodic
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

toml_dict = TOML.parsefile("trj_score.toml")

alpha = toml_dict["param_group"]["alpha"]
beta = toml_dict["param_group"]["beta"]
gamma = toml_dict["param_group"]["gamma"]
sigma = toml_dict["param_group"]["sigma"]
experiment_toml = "Experiment_preprocess_periodic.toml"
generate_score(experiment_toml, alpha,beta,gamma,sigma;res=1)