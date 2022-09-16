abstract type AbstractParams{FT <: AbstractFloat} end

abstract type AbstractModelParams{FT} <: AbstractParams{FT} end
abstract type AbstractOptimizerParams{FT} <: AbstractParams{FT} end
"""
     DataParams{FT} <: AbstractParams{FT}
"""
Base.@kwdef struct DataParams{FT} <: AbstractParams{FT}
    "The batch size"
    batch_size::Int = 64
end

"""
     TrainParams{FT} <: AbstractParams{FT}
"""
Base.@kwdef struct TrainParams{FT} <: AbstractParams{FT}
    "The number of passes, or epochs, over the data to carry out in training"
    nepochs::Int = 100
end
"""
    Parameters{FT,D,O,T,M} <: AbstractParams{FT}

"""
struct Parameters{FT,D,O,T,M} <: AbstractParams{FT}
    "The data hyperparameters"
    data::D
    "The optimizer hyperparameters"
    opt::O
    "The training hyperparameters"
    train::T
    "The model hyperparameters"
    model::M
end

function Parameters{FT}(; data::AbstractParams,
                        train::TrainParams{FT},
                        model::AbstractModelParams{FT},
                        opt::AbstractOptimizerParams{FT}
                        ) where {FT}
    param_args = (data, opt, train, model)
    return Parameters{FT, typeof.(param_args)...}(param_args...)
end

"""
    Args

"""
Base.@kwdef struct Args
    "Type to be used for all computations involving floats"
    FT = Float32
    "Random seed"
    seed = 1
    "A boolean to indicate whether to use GPU or not"
    cuda = true
    "A boolean to indicate whether to compute losses at each epoch"
    compute_losses = true
    "A boolean to indicate whether to checkpoint the model at each epoch"
    checkpointing = true
    "The path to save to and read from if checkpoint or restarting"
    save_path = "examples/mnist/output"
    "A boolean to indicate whether to restart from a checkpoint at `save_path`"
    restart_training = false
end
                    
