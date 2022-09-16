using BSON
using CUDA
using DataLoaders: DataLoader
using Flux
using Flux: params
using FluxTraining
using Images
using Logging: with_logger
using MLDatasets
using MLUtils: shuffleobs, splitobs
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random
using Statistics: mean
using TensorBoardLogger: TBLogger, tb_overwrite

# our package
include("DiffusionModels.jl")

"""
Helper function that loads MNIST images and returns loaders.
"""
function get_data(batch_size, size=32)
    xtrain, _ = MLDatasets.MNIST(:train)[:]
    xtrain = Images.imresize(xtrain, (size, size))
    xtrain = reshape(xtrain, size, size, 1, :)
    xtrain = shuffleobs(xtrain)
    loader_train = DataLoader(xtrain, batch_size)

    xtest, _ = MLDatasets.MNIST(:test)[:]
    xtest = Images.imresize(xtest, (size, size))
    xtest = reshape(xtest, size, size, 1, :)
    loader_test = DataLoader(xtest, batch_size)

    return loader_train, loader_test
end

"""
Helper function from DrWatson.jl to convert a struct to a dict
"""
function struct2dict(::Type{DT}, s) where {DT<:AbstractDict}
    DT(x => getfield(s, x) for x in fieldnames(typeof(s)))
end
struct2dict(s) = struct2dict(Dict, s)

# arguments for the `train` function 
@with_kw struct Args
    η = 1e-4                                        # learning rate
    batch_size = 64                                 # batch size
    nepochs = 100                                   # number of epochs
    ema_rate = 0                                    # exponential moving average rate
    seed = 1                                        # random seed
    cuda = true                                     # use GPU
    compute_losses = true                           # compute losses   
    checkpointing = true                            # use checkpointing
    save_path = "output"                            # results path
    restart_training = false                        # restart training
end

function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
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

    # load data
    loader_train, loader_test = get_data(args.batch_size)

    # model
    if args.restart_training
        model_path = joinpath(args.save_path, "checkpoint_model.bson")
        BSON.@load model_path, model, args
    else
        net = DiffusionModels.NoiseConditionalScoreNetwork()
        model = DiffusionModels.VarianceExplodingSDE(net=net)
    end
    model = model |> device

    # exp moving avg model for storage
    net_ema = DiffusionModels.NoiseConditionalScoreNetwork()
    model_ema = DiffusionModels.VarianceExplodingSDE(net=net_ema)
    model_ema = model_ema |> device

    # loss
    lossfn(x) = DiffusionModels.score_matching_loss(model, x)

    # optimizer
    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(model)
    ps_ema = Flux.params(model_ema)

    # fit the model
    loss_train, loss_test = Inf, Inf
    @info "Start Training, total $(args.nepochs) epochs"
    for epoch = 1:args.nepochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader_train); showspeed=true)

        for batch in loader_train
            # gradient step
            grad = Flux.gradient(() -> lossfn(device(batch)), ps)
            Flux.Optimise.update!(opt, ps, grad)

            # exp moving avg update
            # ps_ema .= @. ps_ema * args.ema_rate + (1 - args.ema_rate) * ps

            next!(progress)
        end

        if args.compute_losses
            loss_train = mean(map(lossfn ∘ device, loader_train))
            loss_test = mean(map(lossfn ∘ device, loader_test))
            @info "Train loss: $(loss_train), Test loss: $(loss_test)"
        end

        if args.checkpointing
            model_path = joinpath(args.save_path, "checkpoint_model.bson")
            let model = cpu(model), args = struct2dict(args)
                BSON.@save model_path model args
                @info "Model saved: $(model_path)"
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end