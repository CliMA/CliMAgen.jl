using BSON
using CUDA
using DataLoaders: DataLoader
using Flux
using Flux: params
using Images
using Logging: with_logger
using MLDatasets
using MLUtils: shuffleobs, splitobs
using ProgressMeter: Progress, next!
using Random
using Statistics: mean

# our package
using CliMAgen


"""
Helper function from DrWatson.jl to convert a struct to a dict
"""
function struct2dict(::Type{DT}, s) where {DT<:AbstractDict}
    DT(x => getfield(s, x) for x in fieldnames(typeof(s)))
end
struct2dict(s) = struct2dict(Dict, s)


"""
Helper function that loads CIFAR10 images and returns loaders.
"""
function get_data(hptrain, size=32)
    batch_size = hptrain.batch_size
    xtrain, _ = MLDatasets.CIFAR10(:train)[:]
    xtrain = Images.imresize(xtrain, (size, size))
    xtrain = reshape(xtrain, size, size, 1, :)
    xtrain = shuffleobs(xtrain)
    loader_train = DataLoader(xtrain, batch_size)
    # note Song does a "random flip"
    # https://github.com/yang-song/score_sde/blob/main/configs/default_cifar10_configs.py#L44
    xtest, _ = MLDatasets.CIFAR10(:test)[:]
    xtest = Images.imresize(xtest, (size, size))
    xtest = reshape(xtest, size, size, 1, :)
    loader_test = DataLoader(xtest, batch_size)

    return loader_train, loader_test
end

function reinitialize(save_path)
    model_path = joinpath(save_path, "checkpoint_model.bson")
    BSON.@load model_path, model, opt, hp
    return (; opt = opt, model = model, hp = hp)
end

function initialize(hpmodel, hpopt)
    net = CliMAgen.NoiseConditionalScoreNetwork(; inchannels = 3) # does this mean to make the model we need to know about hparams.data?
    model = CliMAgen.VarianceExplodingSDE(;hpmodel = hpmodel, net=net)
    opt = create_optimizer(hpopt)
    return (; opt = opt, model = model)
end

function update_step!(ps, data_loader, lossfn, opt, device)
    progress = Progress(length(data_loader); showspeed=true)
    for batch in data_loader
        grad = Flux.gradient(() -> lossfn(device(batch)), ps)
        Flux.Optimise.update!(opt, ps, grad)
        next!(progress)
    end
end

function compute_losses(loader_train, loader_test, lossfn, device)
        loss_train = mean(map(lossfn ∘ device, loader_train))
        loss_test = mean(map(lossfn ∘ device, loader_test))
        @info "Train loss: $(loss_train), Test loss: $(loss_test)"
end

function checkpoint_step(model, opt, hp, save_path)
        model_path = joinpath(save_path, "checkpoint_model.bson")
        let model = cpu(model)
            BSON.@save model_path model opt hp
            @info "Model saved: $(model_path)"
        end
end

function train(args, hp)
    args.seed > 0 && Random.seed!(args.seed)
    
    # directory structure
    !ispath(args.save_path) && mkpath(args.save_path)
    
    # device config
    if args.cuda && CUDA.has_cuda()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end


    # Load data
    loader_train, loader_test = get_data(hp.data)

    # Load or initialize models and optimizer
    if args.restart_training
        @info "Initializing the model, optimizer, and hp from checkpoint."
        @info "Overwriting passed hyper parameters with checkpoint values."
        (; opt, model, hp)  = reinitialize(args.save_path)
    else
        (; opt, model)  = initialize(hp.model, hp.opt)
    end

    # push model to device
    model = model |> device
    
    # create parameters of model
    ps = Flux.params(model)

    # create the loss function for the model
    lossfn(x) = CliMAgen.score_matching_loss(model, x) 

    # Run training and checkpointing
    loss_train, loss_test = Inf, Inf
    @info "Start Training, total $(hp.train.nepochs) epochs"
    for epoch = 1:hp.train.nepochs
        @info "Epoch $(epoch)"

        update_step!(ps, loader_train, lossfn, opt, device)
        
        if args.compute_losses
            compute_losses(loader_train, loader_test, lossfn, device)
        end
        
        if args.checkpointing
            checkpoint_step(model, opt, hp, args.save_path)
        end
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    # set arguments for run
    args = Args()
    # Make hyperparameters structs
    FT = args.FT
    hpdata = DataParams{FT}(batch_size = 10)
    hptrain = TrainParams{FT}(nepochs = 1)
    hpopt = AdamOptimizerParams{FT}(lr = 2e-4, gradclip = 1.0, n_warmup_steps = 5000)
    hpmodel = VarianceExplodingSDEParams{FT}()
    hp = Parameters{FT}(; data = hpdata, train = hptrain, opt = hpopt, model = hpmodel)

    
    train(args, hp)
end
