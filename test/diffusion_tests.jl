using CliMAgen

using Flux
using Test

@testset "Models" begin
    hpmodel = VarianceExplodingSDEParams{Float32}(10.0)
    m = CliMAgen.VarianceExplodingSDE(;hpmodel=hpmodel, net=(x, t) -> x)

    # test constructor
    @test m.σ_max == 10.0f0

    # test drift
    t = rand(3, 10)
    @test CliMAgen.drift(m, t) == zero(t)

    # test diffusion
    t = rand(3, 10)

    @test CliMAgen.diffusion(m, t) == @. m.σ_max^t 

    # test marginal_prob
    x, t = rand(3, 10), rand(10)
    μ, σ = CliMAgen.marginal_prob(m, x, t)
    @test μ == x
    @test σ == CliMAgen.expand_dims(sqrt.((m.σ_max.^(2 .*t).-1) ./(2*log(m.σ_max))), ndims(x) - 1)
  
    # test dummy score
    x, t = rand(3, 3), rand()
    _, σ = CliMAgen.marginal_prob(m, x, t)  
    @test CliMAgen.score(m, x, t) == x ./ σ
    x, t = rand(11), rand(11)
    _, σ = CliMAgen.marginal_prob(m, x, t)  
    @test CliMAgen.score(m, x, t) == x ./ σ
    x, t = rand(3, 3, 4, 11), rand(11)
    _, σ = CliMAgen.marginal_prob(m, x, t)  
    @test CliMAgen.score(m, x, t) == x ./ σ

    # test forward_sde
    f, g = CliMAgen.forward_sde(m)
    x, t = rand(11), rand(11)
    @test all(f(x, nothing, t) .== CliMAgen.drift(m, t))
    @test all(g(x, nothing, t) .== CliMAgen.diffusion(m, t))

    # test reverse_sde
    f, g = CliMAgen.reverse_sde(m)
    x, t = rand(3), rand(3)
    @test f(x, nothing, t) == CliMAgen.drift(m, t) .- CliMAgen.diffusion(m, t) .^ 2 .* CliMAgen.score(m, x, t)
    @test all(g(x, nothing, t) .== CliMAgen.diffusion(m, t))

    # test reverse_ode
    f = CliMAgen.reverse_ode(m)
    x, t = rand(3), rand(3)
    @test f(x, nothing, t) == CliMAgen.drift(m, t) .- CliMAgen.diffusion(m, t) .^ 2 .* CliMAgen.score(m, x, t) ./ 2
end

@testset "Losses" begin
    hpmodel = VarianceExplodingSDEParams{Float32}()
    m = CliMAgen.VarianceExplodingSDE(;hpmodel = hpmodel, net=(x, t) -> x)
    x = rand(3, 7, 8, 3, 25)
    @test CliMAgen.score_matching_loss(m, x) isa Real
end

@testset "Networks" begin
    net = CliMAgen.NoiseConditionalScoreNetwork(inchannels=3)
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
