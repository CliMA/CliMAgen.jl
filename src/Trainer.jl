#module Trainer
# Coach - start here
# Coach.training_step! = GAN_train_step!()
# train step function
# Optimizer + parameters
# Batch size, epochs
# Logging info (uses callback function)
# Checkpoint info (uses callback function)
using Flux.Optimise: AbstractOptimiser

abstract type AbstractTrainer end
abstract type AbstractCallback end

struct Diagnostics{O} <:AbstractCallback
    diagnostic_function::Function
    saveat::Vector{Int} # 0:10:20:30:
    output::O
end

function Diagnostics(diagnostic_function::Function, saveat::Vector{Int}, savetype::ST) where {ST}
    output = Vector{ST}(undef,length(saveat))
    return Diagnostics{typeof(output)}(diagnostic_function, saveat, output)
end

struct Trainer{N,M,P,O<:AbstractOptimiser,L<:AbstractCallback} <: AbstractTrainer
    optimizers::NTuple{N,O}
    parameters::P
    callbacks::NTuple{M,L}
end

function (trainer::Trainer)(model, data_loader)
    if nepochs in trainer.parameters
        nepochs = trainer.parameters.nepochs
    elseif nbatches in trainer.parameters
        nepochs =  div(parameters.nbatches, length(data_loader))
    else
        error("You need to specify nbatches or nepochs in trainer.parameters.")
    end

    for epoch in nepochs
        for (batch_idx, batch) in enumerate(data_loader)
            train_step!(trainer.optimizers..., model, batch)
        end
        
        for cb in trainer.callbacks
            if batch_idx in cb.saveat
                cb_idx = findfirst(isequal(batch_idx), cb.saveat)
                cb.savedvalues[cb_idx] = cb.diagnostic_function(model, batch)
            end
        end
    end
end

#end