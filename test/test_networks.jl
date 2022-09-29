@testset "ResnetBlock" begin
    # constructor
    channels = 6 => 12
    nspatial = 2
    nembed = 128
    resnetblock = ResnetBlock(channels, nspatial, nembed)
    @test resnetblock isa ResnetBlock

    x = randn(FT, 12, 12, 6, 21)
    tembed = randn(FT, 128, 21)
    @test size(resnetblock(x, tembed)) == (12, 12, 12, 21)
    @test eltype(resnetblock(x, tembed)) == FT
end

@testset "AttentionBlock" begin
    # constructor
    channels = 32 => 32
    nspatial = 2
    attnblock = AttentionBlock(channels, nspatial)
    @test attnblock isa AttentionBlock

    x = randn(FT, 12, 12, 32, 21)
    @test size(attnblock(x)) == (12, 12, 32, 21)
    @test eltype(attnblock(x)) == FT
end

@testset "NCSN Network" begin
    net = CliMAgen.NoiseConditionalScoreNetwork(inchannels=3)
    ps = Flux.params(net)
    k = 5
    x = rand(Float32, 2^k, 2^k, 3, 11)
    t = rand(Float32)
    # forward pass
    @test net(x, t) |> size == size(x)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, t) .^ 2)
    end
    @test loss isa Real

    if CUDA.has_cuda()
        net = CliMAgen.NoiseConditionalScoreNetwork(inchannels=3) |> Flux.gpu
        ps = Flux.params(net)
        k = 5
        x = rand(Float32, 2^k, 2^k, 3, 11) |> Flux.gpu
        t = rand(Float32) |> Flux.gpu
        # forward pass
        @test net(x, t) |> Flux.cpu |> size == size(x |> Flux.cpu)
        # backward pass of dummy loss
        loss, grad = Flux.withgradient(ps) do
            sum(net(x, t) .^ 2) |> Flux.cpu
        end
        @test loss isa Real
    end    
end

 @testset "DDPM Network" begin
    net = CliMAgen.DenoisingDiffusionNetwork(inchannels=3)
    ps = Flux.params(net)
    k = 5
    x = rand(Float32, 2^k, 2^k, 3, 11)
    t = rand(Float32)
    # forward pass
    @test net(x, t) |> size == size(x)
    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, t) .^ 2)
    end
    @test loss isa Real

    if CUDA.has_cuda()
        net = CliMAgen.DenoisingDiffusionNetwork(inchannels=3) |> Flux.gpu
        ps = Flux.params(net)
        k = 5
        x = rand(Float32, 2^k, 2^k, 3, 11) |> Flux.gpu
        t = rand(Float32) |> Flux.gpu
        # forward pass
        @test net(x, t) |> Flux.cpu |> size == size(x |> Flux.cpu)
        # backward pass of dummy loss
        loss, grad = Flux.withgradient(ps) do
            sum(net(x, t) .^ 2) |> Flux.cpu
        end
        @test loss isa Real
    end
end

@testset "NCSN Variant Network" begin
    net = CliMAgen.NoiseConditionalScoreNetworkVariant(inchannels=3)
    ps = Flux.params(net)
    k = 5
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