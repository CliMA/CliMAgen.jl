@testset "ResnetBlockDDN" begin
    # constructor
    channels = 6 => 12
    nspatial = 2
    nembed = 128
    resnetblock = ResnetBlockDDN(channels, nspatial, nembed)
    @test resnetblock isa ResnetBlockDDN

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

@testset "NCSN" begin
    net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=2, outer_kernelsize=5, channels=[32, 64, 128, 256])
    @test size(Flux.params(net.layers.tconv2)[1]) == (5, 5, 128, 32)

    ps = Flux.params(net)
    k = 5
    x = rand(Float32, 2^k, 2^k, 2, 11)
    c=nothing
    t = rand(Float32)
    # forward pass
    @test net(x, c, t) |> size == size(x)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, c, t) .^ 2)
    end
    @test loss isa Real

    shift_input_net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=2, shift_input=true, outer_kernelsize=5,)
    Flux.loadmodel!(shift_input_net, net)
    @test net(x.-mean(x, dims = (1,2)), c, t) ≈ shift_input_net(x, c, t)

    shift_output_net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=2, shift_output=true, outer_kernelsize=5,)
    Flux.loadmodel!(shift_output_net, net)
    @test (net(x, c, t) .-mean(net(x,c, t), dims = (1,2))) ≈ shift_output_net(x, c, t)

    mean_bypass_net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=2, shift_input=true, shift_output=true, mean_bypass=true)
    ps = Flux.params(net)
    k = 5
    x = rand(Float32, 2^k, 2^k, 2, 11)
    t = rand(Float32)
    # forward pass
    @test  mean_bypass_net(x, c, t) |> size == size(x)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, c, t) .^ 2)
    end
    @test loss isa Real
end


@testset "NCSN in and out channels" begin
    # with context
    net = CliMAgen.NoiseConditionalScoreNetwork(context=true, noised_channels=2, context_channels = 3)
    ps = Flux.params(net)
    k = 5
    x = rand(Float32, 2^k, 2^k, 2, 11)
    c = rand(Float32, 2^k, 2^k, 3, 11)
    t = rand(Float32)
    # forward pass
    @test net(x, c, t) |> size == (2^k, 2^k, 2, 11)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, c, t) .^ 2)
    end
    @test loss isa Real

    # once without context (default is context=false and context_channels = 0)
    net2 = CliMAgen.NoiseConditionalScoreNetwork(noised_channels=3)
    ps = Flux.params(net2)
    k = 5
    x = rand(Float32, 2^k, 2^k, 3, 11)
    t = rand(Float32)
    # forward pass
    @test net2(x, nothing, t) |> size == (2^k, 2^k, 3, 11)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net2(x, c, t) .^ 2)
    end
    @test loss isa Real
end
