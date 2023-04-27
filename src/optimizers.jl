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

WarmupSchedule{FT}(n_warmup_steps=1) where {FT} =
    WarmupSchedule(FT(1.0), n_warmup_steps, 0)

function Flux.Optimise.apply!(o::WarmupSchedule{FT}, x, Δ) where {FT}
    η, n_warmup_steps = o.η, o.n_warmup_steps
    current_step = o.current = o.current + 1
    Δ .*= min(η * FT(current_step / n_warmup_steps), η)
end

"""
    ExponentialMovingAverage{FT} <: Flux.Optimisers.AbstractOptimiser

A Flux.Optimisers Optimiser for an exponential moving average accumulation of
parameters during training. The 'rate' parameter is the exponential decay rate.
"""
mutable struct ExponentialMovingAverage{FT} <: Flux.Optimise.AbstractOptimiser
    rate::FT
end

"""
    Flux.update!(opt::ExponentialMovingAverage, ps_smooth::Flux.Params, ps::Flux.Params)

Updates the exponential-moving-average parameters in place.
"""
function Flux.update!(opt::ExponentialMovingAverage, ps_smooth::Flux.Params, ps::Flux.Params)
    for (xs, x) in zip(ps_smooth, ps)
        @. xs = opt.rate * xs + (1 - opt.rate) * x
    end
end
