"""
    ClimaGen.train!
"""
function train!(model, lossfn, dataloaders, opt, opt_smooth, nepochs, device::Function; savedir="./output/", logger=nothing, freq_chckpt=Inf)
    # model parameters
    ps = Flux.params(model)

    # setup smoothed parameters & model
    model_smooth = deepcopy(model)
    ps_smooth = Flux.params(model_smooth)

    # training loop
    loss_train, loss_test = Inf, Inf
    @info "Start Training, total $(nepochs) epochs"
    for epoch = 1:nepochs
        @info "Epoch $(epoch)"

        # update the params and compute losses
        CliMAgen.update_step!(ps, ps_smooth, opt, opt_smooth, dataloaders.loader_train, lossfn, device)
        loss_train, loss_test = CliMAgen.compute_losses(lossfn, dataloaders, device)
        for (ltrain, ltest) in zip(loss_train, loss_test)
            @info "Loss: $(ltrain) (train) | $(ltest) (test)"
        end

        # store model checkpoint on disk
        savepath = joinpath(savedir, "checkpoint.bson")
        CliMAgen.save_model_and_optimizer(Flux.cpu(model), Flux.cpu(model_smooth), opt, opt_smooth, savepath)
        @info "Checkpoint saved to $(savepath)."
        if !(logger isa Nothing)
            log_dict(
                logger, 
                Dict(
                    ["Training/Loss$i" => l for (i,l) in enumerate(loss_train)]...,
                    ["Testing/Loss$i" => l for (i,l) in enumerate(loss_test)]...,
                )
            ) 
        end

        # every so often, we upload a checkpoint to the cloud if applicable
        if epoch % freq_chckpt == 0
            if !(logger isa Nothing)
                log_artifact(logger, savepath; name="checkpoint-$(epoch)", type="BSON-file")
            end
        end
    end
end

"""
    ClimaGen.update_step!
"""
function update_step!(ps, ps_smooth, opt, opt_smooth, loader_train, lossfn::Function, device::Function)
    progress = ProgressMeter.Progress(length(loader_train); showspeed=true)

    # epoch loop
    for batch in loader_train
        batch = device(batch)
        
        grad = Flux.gradient(() -> sum(lossfn(batch)), ps)

        Flux.Optimise.update!(opt, ps, grad)
        Flux.Optimise.update!(opt_smooth, ps_smooth, ps)

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
function save_model_and_optimizer(model, model_smooth, opt, opt_smooth, path::String)
    BSON.@save path model model_smooth opt opt_smooth
    @info "Model saved at $(path)."
end

"""
    ClimaGen.load_model_and_optimizer
"""
function load_model_and_optimizer(path::String)
    BSON.@load path model model_smooth opt opt_smooth
    return (; model=model, model_smooth=model_smooth, opt=opt, opt_smooth=opt_smooth)
end
