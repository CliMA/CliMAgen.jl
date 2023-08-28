module ScoreGenMod

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

export generate_score

function generate_score(beta,gamma,sigma)
    package_dir = pkgdir(CliMAgen)
    include(joinpath(package_dir,"examples/utils_data.jl"))
    include(joinpath(package_dir,"examples/utils_analysis.jl"))

    experiment_toml = "correlated_ou/Experiment.toml"
    FT = Float32
    logger=nothing

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction = params.data.fraction
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

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

    @info "The first line"
    # set up device
    if !nogpu && CUDA.has_cuda()
        device = Flux.gpu
        array_type = CuArray
        @info "Sampling on GPU"
    else
        device = Flux.cpu
        array_type = Array
        @info "Sampling on CPU"
    end
    f_path = "correlated_ou/data/data_$(beta)_$(gamma)_$(sigma).hdf5"
    f_variable = "timeseries"
    # set up dataset
    @info "creating data loader"
    dataloaders = get_data(
        f_path, f_variable,batchsize;
        f = fraction,
        FT=Float32,
        rng=Random.GLOBAL_RNG
    )

    # note we are assuming that we run everything from examples directory
    savedir = pwd() * "/correlated_ou/output"
    # set up model and optimizers
    if isfile(savedir * "/checkpoint_$(beta)_$(gamma)_$(sigma).bson")
        @info "removing checkpoint"
        rm(savedir * "/checkpoint_$(beta)_$(gamma)_$(sigma).bson") 
    end

    if isfile(savedir * "/checkpoint.bson")
        @info "removing checkpoint"
        rm(savedir * "/checkpoint.bson") 
    end

    if isfile(savedir * "/losses_$(beta)_$(gamma)_$(sigma).txt") 
        @info "removing losses"
        rm(savedir * "/losses_$(beta)_$(gamma)_$(sigma).txt") 
    end

    @info "creating checkpoint path"
    println(" at " * savedir)
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    @info "creating loss file"
    loss_file = joinpath(savedir, "losses_$(beta)_$(gamma)_$(sigma).txt")

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
                                       proj_kernelsize = proj_kernelsize,
                                       outer_kernelsize = outer_kernelsize,
                                       middle_kernelsize = middle_kernelsize,
                                       inner_kernelsize = inner_kernelsize
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
        logger = logger,
        freq_chckpt = freq_chckpt,
    )

    @info "moving data"
    mv(checkpoint_path, savedir * "/checkpoint_$(beta)_$(gamma)_$(sigma).bson")

    @info "new checkpiont path"
    checkpoint_path = joinpath(savedir, "checkpoint_$(beta)_$(gamma)_$(sigma).bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)

    @info "writing data"
    hfile = h5open("correlated_ou/data/data_$(beta)_$(gamma)_$(sigma).hdf5")
    data = read(hfile["timeseries"])
    close(hfile) 
    data = data[:,:,1:end]
    times = [1:size(data)[3]...]  
    t0 = Float32(0.)
    N = size(data)[1]
    @info "calculating scores"
    model = device(model)
    Nt  = length(times)
    batchsize = params.data.batchsize
    m = floor(Int, Nt/ batchsize)
    indexlist = UnitRange{Int64}[]
    for i in 1:m+1
        if i == m+1
            push!(indexlist, (i-1)*batchsize+1:Nt)
        else
            push!(indexlist, (i-1)*batchsize+1:i*batchsize)
        end
    end
    scores = zeros(N,N,length(times))
    for indices in ProgressBar(indexlist)
        x_A1 = Float32.(array_type(reshape(data[:,:,indices], (N,N,1,length(indices)))))
        scores1 = CliMAgen.score(model, x_A1, t0)
        scores1 = Array(scores1);
        scores[:,:, indices] = scores1[:,:,1,:] 
    end
    

    @info "writing data part 2"
    f_path = "correlated_ou/data/scores_$(beta)_$(gamma)_$(sigma).hdf5"
    hfile = h5open(f_path,"w")
    write(hfile,"scores",scores)
    write(hfile,"timeseries",data)
    close(hfile)
    @info "done!"
end
end