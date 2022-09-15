using FluxTraining: runstep
using FluxTraining: AbstractTrainingPhase, AbstractValidationPhase
using FluxTraining.Events: LossBegin, BackwardBegin, BackwardEnd

import FluxTraining

struct Train <: AbstractTrainingPhase end
struct Test <: AbstractValidationPhase end

function FluxTraining.step!(learner, phase::Train, batch)
    (xs, ys) = batch
    runstep(learner, phase, (xs=xs, ys=ys)) do handle, state
        grads = gradient(learner.params) do
            handle(LossBegin())
            println(xs |> typeof)
            loss = learner.lossfn(xs, ys)
            state.loss = loss
            handle(BackwardBegin())
            return loss
        end
        state.grads = grads
        handle(BackwardEnd())
        Flux.Optimise.update!(learner.optimizer, learner.params, grads)
    end
end

function FluxTraining.step!(learner, phase::Test, batch)
    (xs, ys) = batch
    runstep(learner, phase, (xs=xs, ys=ys)) do _, state
        state.loss = learner.lossfn(learner.model, xs, ys)
    end
end