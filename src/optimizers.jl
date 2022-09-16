"""
    WarmupSchedule{FT} <: Flux.Optimisers.AbstractOptimiser

An Flux.Optimisers Optimiser for a warmup schedule, where
the gradient multiplier varies from 1.0/n_warmup_steps
to 1.0 linearly over n_warmup_steps
- not over epochs! - and remains constant at 1.0 afterwards.

Note that this optimizer is intended for use with an optimizer like Adam - the scale
of the learning rate is set by the Adam optimizer's learning rate parameter.

The default n_warmup_steps is 1, which would imply *no* warmup.
"""
mutable struct WarmupSchedule{FT} <: Flux.Optimise.AbstractOptimiser
    η::FT
    n_warmup_steps::Int
    current::Int
end

WarmupSchedule{FT}(n_warmup_steps = 1) where{FT} =
  WarmupSchedule(FT(1.0), n_warmup_steps, 0)

function Flux.Optimise.apply!(o::WarmupSchedule{FT}, x, Δ) where{FT}
    η, n_warmup_steps = o.η, o.n_warmup_steps
    current_step = o.current = o.current+1
    @. Δ *= min(η * FT(current_step/n_warmup_steps), η)
end

"""
    struct OptimizerHyperParams{FT}

All hyperparameters needed for constructing an Adam optimizer,
optionally with gradient clipping and with a warmup schedule.

If gradclip is positive, the gradient clip is applied via the L2
norm using a value of `gradclip`. If gradclip is 0 or negative,
no clipping is applied.

If n_warmup_steps is 1, no warmup schedule is implemented.

The defaults are *no* clipping and *no* warmup.
"""
Base.@kwdef struct OptimizerHyperParams{FT}
    lr = FT(0.0002)
    ϵ = FT(1e-8)
    gradclip = FT(0.0)
    β1 = FT(0.9)
    β2 = FT(0.999)
    n_warmup_steps::Int = 1
end
"""
    create_optimizer(hparams::OptimizerHyperParams{FT}) where {FT}

Create and return the optimizer desired by composing (1) gradient clipping, (2)
a warmup schedule for the learning rate multiplier, and (3) the Adam optimizer,
based on the hyperparams `hparam` passed.
"""
function create_optimizer(hparams::OptimizerHyperParams{FT}) where {FT}
    if hparams.gradclip >eps(FT)
        opt = Flux.Optimise.Optimiser(Flux.Optimise.ClipNorm(hparams.gradclip),
                                      WarmupSchedule{FT}(hparams.n_warmup_steps),
                                      Flux.Optimise.Adam(hparams.lr, (hparams.β1, hparams.β2), hparams.ϵ))
    else
        opt = Flux.Optimise.Optimiser(WarmupSchedule{FT}(hparams.n_warmup_steps),
                                      Flux.Optimise.Adam(hparams.lr, (hparams.β1, hparams.β2), hparams.ϵ))
    end
    
    return opt
end


