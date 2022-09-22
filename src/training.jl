"""
    ClimaGen.train!
"""
function train!(model, lossfn, dataloaders, opt, hparams::HyperParameters, args::NamedTuple, device::Function, logger=nothing)
    # unpack the relevant parameters
    nepochs, = hparams.training
    savepath = joinpath(args.savedir, "checkpoint.bson")

    # update the logger config with hyperparms
    if logger isa AbstractLogger
        log_config(logger, CliMAgen.struct2dict(hparams))
    end

    # model parameters
    ps = Flux.params(model)

    # training loop
    loss_train, loss_test = Inf, Inf
    @info "Start Training, total $(nepochs) epochs"
    for epoch = 1:nepochs
        @info "Epoch $(epoch)"

        # update the params and compute losses
        CliMAgen.update_step!(ps, lossfn, dataloaders.loader_train, opt, device)
        loss_train, loss_test = CliMAgen.compute_losses(lossfn, dataloaders, device)
        @info "Train loss: $(loss_train), Test loss: $(loss_test)"

        # store model checkpoint on disk
        CliMAgen.save_model_and_optimizer(Flux.cpu(model), opt, hparams, savepath)

        # log training and testing loss
        if logger isa AbstractLogger
            log_dict(logger, Dict("Training/Loss" => loss_train, "Testing/Loss" => loss_test))
        end
    end

    # after training is complete, log last model checkpoint
    if logger isa AbstractLogger
        log_artifact(logger, savepath)
    end
end

"""
    ClimaGen.update_step!
"""
function update_step!(ps, lossfn, loader_train, opt, device::Function)
    # set up progress bafr
    progress = ProgressMeter.Progress(length(loader_train); showspeed=true)

    # epoch loop
    for batch in loader_train
        batch = device(batch)
        grad = Flux.gradient(() -> lossfn(batch), ps)
        Flux.Optimise.update!(opt, ps, grad)
        ProgressMeter.next!(progress)
    end
end

"""
    ClimaGen.compute_losses
"""
function compute_losses(lossfn, dataloaders, device::Function)
    loader_train, loader_test, = dataloaders
    loss_train = Statistics.mean(map(lossfn ∘ device, loader_train))
    loss_test = Statistics.mean(map(lossfn ∘ device, loader_test))

    return loss_train, loss_test
end

"""
    ClimaGen.save_model_and_optimizer
"""
function save_model_and_optimizer(model, opt, hparams::HyperParameters, path::String)
    BSON.@save path model opt hparams
    @info "Model saved at $(path)."
end

"""
    ClimaGen.load_model_and_optimizer
"""
function load_model_and_optimizer(path::String)
    BSON.@load path model opt hparams
    return (; model=model, opt=opt, hparams=hparams)
end
