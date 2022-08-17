using Downscaling
using Flux
using Test
using Random

FT = Float32

@testset "Downscaling.jl" begin
    # PatchBlock
    in_channels = 3
    out_channels = 7
    img_size = 97
    batch_size = 13
    stride = 2
    patch_block = PatchBlock(in_channels, out_channels, stride=stride)
    x = randn(FT, (img_size, img_size, in_channels, batch_size))
    @test patch_block(x) |> size == (48, 48, 7, 13)

    in_channels = 3
    out_channels = 7
    img_size = 97
    batch_size = 13
    stride = 2
    patch_block = PatchBlock1D(in_channels, out_channels, stride=stride)
    x = randn(FT, (img_size, in_channels, batch_size))
    @test patch_block(x) |> size == (48, 7, 13)

    # PatchDiscriminator
    in_channels = 3
    batch_size = 13
    patch_dsicriminator = PatchDiscriminator(in_channels)
    x = randn(FT, (img_size, img_size, in_channels, batch_size))
    @test patch_dsicriminator(x) |> size == (10, 10, 1, 13)

    in_channels = 3
    batch_size = 13
    patch_dsicriminator = PatchDiscriminator1D(in_channels)
    x = randn(FT, (img_size, in_channels, batch_size))
    @test patch_dsicriminator(x) |> size == (10, 1, 13)

    # ConvBlock
    kernel_size = 3
    in_channels = 3
    out_channels = 7
    with_activation = true
    down = true
    batch_size = 13
    img_size = 97
    conv_block = ConvBlock(kernel_size, in_channels, out_channels, with_activation, down)
    x = randn(FT, (img_size, img_size, in_channels, batch_size))
    @test conv_block(x) |> size == (95, 95, 7, 13)

    kernel_size = 3
    in_channels = 3
    out_channels = 7
    with_activation = true
    down = false
    batch_size = 13
    img_size = 97
    conv_block = ConvBlock(kernel_size, in_channels, out_channels, with_activation, down)
    x = randn(FT, (img_size, img_size, in_channels, batch_size))
    @test conv_block(x) |> size == (99, 99, 7, 13)

    kernel_size = 3
    in_channels = 3
    out_channels = 7
    with_activation = true
    down = true
    batch_size = 13
    img_size = 97
    conv_block = ConvBlock1D(kernel_size, in_channels, out_channels, with_activation, down)
    x = randn(FT, (img_size, in_channels, batch_size))
    @test conv_block(x) |> size == (95, 7, 13)

    kernel_size = 3
    in_channels = 3
    out_channels = 7
    with_activation = true
    down = false
    batch_size = 13
    img_size = 97
    conv_block = ConvBlock1D(kernel_size, in_channels, out_channels, with_activation, down)
    x = randn(FT, (img_size, in_channels, batch_size))
    @test conv_block(x) |> size == (99, 7, 13)

    # ResidualBlock
    in_channels = 3
    batch_size = 13
    img_size = 97
    residual_block = ResidualBlock(in_channels)
    x = randn(FT, (img_size, img_size, in_channels, batch_size))
    @test residual_block(x) |> size == (97, 97, 3, 13)

    in_channels = 3
    batch_size = 13
    img_size = 97
    residual_block = ResidualBlock1D(in_channels)
    x = randn(FT, (img_size, in_channels, batch_size))
    @test residual_block(x) |> size == (97, 3, 13)

    # UNetGenerator
    img_size = 128
    batch_size = 5
    in_channels = 1
    num_features = 64
    num_residual = 9
    unet = UNetGenerator(in_channels, num_features, num_residual)
    x = randn(FT, (img_size, img_size, in_channels, batch_size))
    @test unet(x) |> size == (img_size, img_size, in_channels, batch_size)

    img_size = 128
    batch_size = 5
    in_channels = 1
    num_features = 64
    num_residual = 8
    unet = NoisyUNetGenerator(in_channels, num_features, num_residual)
    x = randn(FT, (img_size, img_size, in_channels, batch_size))
    noise = 2 .* rand(FT, (div(img_size, 4), div(img_size, 4), num_features * 4, batch_size)) .- 1
    y = unet(x, noise)
    @test unet(x, noise) |> size == (img_size, img_size, in_channels, batch_size)
    @test_throws AssertionError NoisyUNetGenerator(in_channels, num_features, 3)

    img_size = 128
    batch_size = 5
    in_channels = 1
    num_features = 64
    num_residual = 9
    unet = UNetGenerator1D(in_channels, num_features, num_residual)
    x = randn(FT, (img_size, in_channels, batch_size))
    @test unet(x) |> size == (img_size, in_channels, batch_size)

    # img_size = 128
    # batch_size = 5
    # in_channels = 2
    # num_features = 64
    # num_residual = 9
    # unet = UNetGeneratorAR(in_channels, num_features, num_residual)
    # x = randn(FT, (img_size, img_size, in_channels, batch_size))
    # @test unet(x) |> size == (img_size, img_size, in_channels, batch_size)

    # img_size = 128
    # batch_size = 5
    # in_channels = 4
    # num_features = 64
    # num_residual = 9
    # unet = UNetGeneratorAR(in_channels, num_features, num_residual)
    # x = randn(FT, (img_size, img_size, in_channels, batch_size))
    # @test unet(x) |> size == (img_size, img_size, in_channels, batch_size)

    # Patchnet
    img_size = 128
    batch_size = 5
    in_channels = 1
    num_features = 64
    num_residual = 9
    net = PatchNet(in_channels, num_features, num_residual)
    x = randn(FT, (img_size, img_size, in_channels, batch_size))
    @test net(x) |> size == (img_size / 2, img_size / 2, in_channels, batch_size)
end
