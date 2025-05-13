"""
    train!(model,
           model_smooth,
           lossfn,
           dataloaders,
           opt,
           opt_smooth,
           nepochs,
           device::Function;
           start_epoch=1,
           savedir="./output/",
           logger=nothing,
           freq_chckpt=Inf)

Carries out the training of the diffusion model
`model` by iterating over all batches of the training data `nepochs` times.

Notes:
- The `model` parameters are updated via gradient descent of `lossfn`, 
  while the `model_smooth` parameters are updated using an exponential
  moving average. 
- After each epoch, the models are saved, and the loss is computed on both 
  the training and test data and saved in `savedir`.
- If restarting from a previous run, start_epoch is no longer 1, but the final 
  epoch is still `nepochs`.
- The training and test data are stored in `dataloaders`.
"""
function train!(model,
                model_smooth,
                lossfn,
                dataloaders,
                opt,
                opt_smooth,
                nepochs,
                device::Function;
                start_epoch=1,
                savedir="./output/",
                logger=nothing,
                freq_chckpt=Inf)
    # training loop
    loss_train, loss_test = Inf, Inf
    @info "Start Training, total $(nepochs) epochs"
    for epoch = start_epoch:nepochs
        @info "Epoch $(epoch)"

        # update the params and compute losses
        CliMAgen.update_step!(opt, opt_smooth, model, model_smooth, dataloaders.loader_train, lossfn, device)
        loss_train, loss_test = CliMAgen.compute_losses((x) -> lossfn(model, x), dataloaders, device)
        for (ltrain, ltest) in zip(loss_train, loss_test)
            @info "Loss: $(ltrain) (train) | $(ltest) (test)"
        end
        open(joinpath(savedir, "losses.txt"),"a") do io
            DelimitedFiles.writedlm(io, transpose([epoch, loss_train...,loss_test...]),',')
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

        # every so often, we save a checkpoint
        if epoch % freq_chckpt == 0
            savepath = joinpath(savedir, "checkpoint_$(epoch).bson")
            CliMAgen.save_model_and_optimizer(Flux.cpu(model), Flux.cpu(model_smooth), opt, opt_smooth, savepath)
            @info "Checkpoint saved at Epoch $(epoch)."
        end
    end
end

"""
    CliMAgen.update_step!(ps, ps_smooth, opt, opt_smooth, loader_train, lossfn::Function, device::Function)

Updates the parameters `ps` and the exponential-moving averaged set `ps_smooth) by computing
a gradient of the `lossfn` evaluated on each batch of `loader_train`. 
"""
function update_step!(opt, opt_smooth, model, model_smooth, loader_train, lossfn::Function, device::Function)
    progress = ProgressMeter.Progress(length(loader_train); showspeed=true)
    opt_state = Flux.setup(opt, model)
    #opt_smooth_state = Flux.setup(opt_smooth, model_smooth)
    # epoch loop
    for batch in loader_train
        batch = device(batch)
        
        grad = Flux.gradient((m) -> sum(lossfn(m, batch)), model);

        Flux.update!(opt_state, model, grad[1]); # updates opt_state and model in place
        Flux.update!(opt_smooth, model_smooth, model) # this doesnt require the opt_state, is there a better way?

        ProgressMeter.next!(progress)
    end
end

"""
    CliMAgen.compute_losses(lossfn, dataloaders, device::Function)

Computes and returns the value of the `lossfn` on the training
and test data stored in `dataloaders`. The computation
is carried out on the GPU or CPU according to `device`.
"""
function compute_losses(lossfn, dataloaders, device::Function)
    loader_train, loader_test, = dataloaders
    loss_train = Statistics.mean(map(lossfn ∘ device, loader_train))
    loss_test = Statistics.mean(map(lossfn ∘ device, loader_test))

    return loss_train, loss_test
end

"""
    CliMAgen.save_model_and_optimizer(model, model_smooth, opt, opt_smooth, path::String)

Saves the model parameters and optimizer 
for both the instantaneous model
as well as the exponentially-averaged one to `path`.
"""
function save_model_and_optimizer(model, model_smooth, opt, opt_smooth, path::String)
    BSON.@save path model model_smooth opt opt_smooth
    @info "Model saved at $(path)."
end

"""
    CliMAgen.load_model_and_optimizer(path::String)

Loads and returns the model parameters and optimizer 
for both the instantaneous model
as well as the exponentially-averaged one.
"""
function load_model_and_optimizer(path::String)
    BSON.@load path model model_smooth opt opt_smooth
    return (; model=model, model_smooth=model_smooth, opt=opt, opt_smooth=opt_smooth)
end
