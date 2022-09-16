using Downscaling
using Flux
using Test
using Random
using Statistics:mean 
FT = Float32

@testset "WarmupSchedule" begin
    FT = Float32
    default_warmup = WarmupSchedule{FT}()
    p = [0.0]
    nsteps = 120
    steps = 1:nsteps
    grads = []
    grad = [1.0]
    for _ in steps
        grad .= [1.0]
        Flux.Optimise.apply!(default_warmup, p, grad)
        push!(grads, grad[1])
    end
    @test length(unique(grads)) == 1
    @test unique(grads)[1] == FT(1.0)
    @test default_warmup.current == nsteps
    
    n_warmup_steps = 100
    linear_warmup = WarmupSchedule{FT}(n_warmup_steps)
    grads = []
    grad = [1.0]
    for i in 1:nsteps
        grad .= [1.0]
        Flux.Optimise.apply!(linear_warmup, p, grad)
        push!(grads, grad[1])
    end
    @test linear_warmup.current == nsteps
    @test grads ≈ [min(FT(1.0), FT(n/n_warmup_steps)) for n in 1:nsteps]
    
    
    opt = WarmupSchedule{FT}(n_warmup_steps)
    p = [FT(1.0)]
    x = FT.(randn(100))
    loss(x) = mean((p.*x).^FT(2.0))
    actual_grad(θ,x,i) = mean(x.^FT(2.0))*FT(2.0)*θ.*i/n_warmup_steps
    
    p_array = [p[1]]
    for i in 1:nsteps
        θ = Flux.Params([p])
        θ̄ = Flux.gradient(() -> loss(x), θ)
        Flux.Optimise.update!(opt, θ, θ̄)
        push!(p_array, p[1])
    end
    @test (p_array[1:end-1] .- p_array[2:end]) ≈ actual_grad.(p_array[1:end-1], Ref(x),1:nsteps)
    
end


@testset "create_optimizer" begin
    FT = Float32
    hparams = OptimizerHyperParams{FT}(gradclip = FT(1.0))
    opt = create_optimizer(hparams)
    @test opt[1].thresh == hparams.gradclip
    @test opt[2].n_warmup_steps == hparams.n_warmup_steps
    @test opt[3].eta ≈ hparams.lr
    @test opt[3].beta[1] ≈ hparams.β1
    @test opt[3].beta[2] ≈ hparams.β2
    @test opt[3].epsilon ≈ hparams.ϵ
    
    #no gradient clipping here
    hparams = OptimizerHyperParams{FT}()
    opt = create_optimizer(hparams)
    @test opt[1].n_warmup_steps == hparams.n_warmup_steps
    @test opt[2].eta ≈ hparams.lr
    @test opt[2].beta[1] ≈ hparams.β1
    @test opt[2].beta[2] ≈ hparams.β2
    @test opt[2].epsilon ≈ hparams.ϵ
end
