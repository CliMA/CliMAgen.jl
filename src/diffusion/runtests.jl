include("DiffusionModels.jl")

using Flux
using Test

@testset begin
    "Models"
    m = DiffusionModels.VarianceExplodingSDE(net=(x, t) -> x)

    # test constructor
    @test m.σ_min == 0.01f0
    @test m.σ_max == 50.0f0

    # test drift
    t = rand(3, 10)
    @test DiffusionModels.drift(m, t) == zero(t)

    # test diffusion
    t = rand(3, 10)
    @test DiffusionModels.diffusion(m, t) == @. m.σ_min * (m.σ_max / m.σ_min)^t * sqrt(2 * (log(m.σ_max) - log(m.σ_min)))

    # test marginal_prob
    x, t = rand(3, 10), rand(10)
    μ, σ = DiffusionModels.marginal_prob(m, x, t)
    @test μ == x
    @test σ == DiffusionModels.expand_dims(m.σ_min .* (m.σ_max / m.σ_min) .^ t, ndims(x) - 1)
    x, t = rand(3), rand()
    μ, σ = DiffusionModels.marginal_prob(m, x, t)
    @test μ == x
    @test σ == DiffusionModels.expand_dims(m.σ_min .* (m.σ_max / m.σ_min) .^ t, ndims(x) - 1)

    # test dummy score
    x, t = rand(3, 3), rand()
    _, σ = DiffusionModels.marginal_prob(m, x, t)  
    @test DiffusionModels.score(m, x, t) == x ./ σ
    x, t = rand(11), rand(11)
    _, σ = DiffusionModels.marginal_prob(m, x, t)  
    @test DiffusionModels.score(m, x, t) == x ./ σ
    x, t = rand(3, 3, 4, 11), rand(11)
    _, σ = DiffusionModels.marginal_prob(m, x, t)  
    @test DiffusionModels.score(m, x, t) == x ./ σ

    # test forward_sde
    f, g = DiffusionModels.forward_sde(m)
    x, t = rand(11), rand(11)
    @test all(f(x, nothing, t) .== DiffusionModels.drift(m, t))
    @test all(g(x, nothing, t) .== DiffusionModels.diffusion(m, t))

    # test reverse_sde
    f, g = DiffusionModels.reverse_sde(m)
    x, t = rand(3), rand(3)
    @test f(x, nothing, t) == DiffusionModels.drift(m, t) .- DiffusionModels.diffusion(m, t) .^ 2 .* DiffusionModels.score(m, x, t)
    @test all(g(x, nothing, t) .== DiffusionModels.diffusion(m, t))

    # test reverse_ode
    f = DiffusionModels.reverse_ode(m)
    x, t = rand(3), rand(3)
    @test f(x, nothing, t) == DiffusionModels.drift(m, t) .- DiffusionModels.diffusion(m, t) .^ 2 .* DiffusionModels.score(m, x, t) ./ 2
end

@testset begin
    "Losses"
    m = DiffusionModels.VarianceExplodingSDE(net=(x, t) -> x)
    x = rand(3, 7, 8, 3, 25)
    @test DiffusionModels.score_matching_loss(m, x) isa Real
end

@testset begin
    "Networks"
    net = DiffusionModels.NoiseConditionalScoreNetwork(inchannels=3)
    ps = Flux.params(net)
    for k in 5:6
        x = rand(Float32, 2^k, 2^k, 3, 11)
        t = rand(Float32)

        # forward pass
        @test net(x, t) |> size == size(x)

        # backward pass of dummy loss
        loss, grad = Flux.withgradient(ps) do
            sum(net(x, t) .^ 2)
        end
        @test loss isa Real
    end
end