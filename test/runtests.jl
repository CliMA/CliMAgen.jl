using Downscaling
using Flux
using Test
using Random

FT = Float32

@testset "Downscaling.jl" begin
    @testset "PatchBlock" begin
        # PatchBlock 1D
        in_channels = 3
        out_channels = 7
        img_size = 97
        batch_size = 13
        stride = 2
        patch_block = PatchBlock1D(in_channels, out_channels, stride=stride)
        x = randn(FT, (img_size, in_channels, batch_size))
        @test patch_block(x) |> size == (48, 7, 13)

        # PatchBlock 2D
        in_channels = 3
        out_channels = 7
        img_size = 97
        batch_size = 13
        stride = 2
        patch_block = PatchBlock2D(in_channels, out_channels, stride=stride)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test patch_block(x) |> size == (48, 48, 7, 13)
    end

    @testset "ConvBlock" begin
        # ConvBlock 1D
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

        # ConvBlock 2D
        kernel_size = 3
        in_channels = 3
        out_channels = 7
        with_activation = true
        down = true
        batch_size = 13
        img_size = 97
        conv_block = ConvBlock2D(kernel_size, in_channels, out_channels, with_activation, down)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test conv_block(x) |> size == (95, 95, 7, 13)

        kernel_size = 3
        in_channels = 3
        out_channels = 7
        with_activation = true
        down = false
        batch_size = 13
        img_size = 97
        conv_block = ConvBlock2D(kernel_size, in_channels, out_channels, with_activation, down)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test conv_block(x) |> size == (99, 99, 7, 13)
    end

    @testset "ResidualBlock" begin
        # ResidualBlock 1D
        in_channels = 3
        batch_size = 13
        img_size = 97
        residual_block = ResidualBlock1D(in_channels)
        x = randn(FT, (img_size, in_channels, batch_size))
        @test residual_block(x) |> size == (97, 3, 13)

        # ResidualBlock 2D
        in_channels = 3
        batch_size = 13
        img_size = 97
        residual_block = ResidualBlock2D(in_channels)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test residual_block(x) |> size == (97, 97, 3, 13)
    end

    @testset "PatchDiscriminator" begin
        # PatchDiscriminator 1D
        img_size = 100
        in_channels = 3
        batch_size = 13
        patch_dsicriminator = PatchDiscriminator1D(in_channels)
        x = randn(FT, (img_size, in_channels, batch_size))
        @test patch_dsicriminator(x) |> size == (10, 1, 13)

        # PatchDiscriminator 2D
        img_size = 100
        in_channels = 3
        batch_size = 13
        patch_dsicriminator = PatchDiscriminator2D(in_channels)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test patch_dsicriminator(x) |> size == (10, 10, 1, 13)
    end

    @testset "UNetGenerator" begin
        # UNetGenerator 1D
        img_size = 128
        batch_size = 5
        in_channels = 1
        num_features = 64
        num_residual = 9
        unet = UNetGenerator1D(in_channels, num_features, num_residual)
        x = randn(FT, (img_size, in_channels, batch_size))
        @test unet(x) |> size == (img_size, in_channels, batch_size)

        # UNetGenerator 2D
        img_size = 128
        batch_size = 5
        in_channels = 1
        num_features = 64
        num_residual = 9
        unet = UNetGenerator2D(in_channels, num_features, num_residual)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        @test unet(x) |> size == (img_size, img_size, in_channels, batch_size)
    end

    @testset "NoisyUNetGenerator" begin
        # NoisyUNetGenerator1D
        img_size = 128
        batch_size = 5
        in_channels = 1
        num_features = 64
        num_residual = 8
        unet = NoisyUNetGenerator1D(in_channels, num_features, num_residual)
        x = randn(FT, (img_size, in_channels, batch_size))
        noise = 2 .* rand(FT, (div(img_size, 4), num_features * 4, batch_size)) .- 1
        y = unet(x, noise)
        @test unet(x, noise) |> size == (img_size, in_channels, batch_size)
        @test_throws AssertionError NoisyUNetGenerator1D(in_channels, num_features, 3)

        # NoisyUNetGenerator2D
        img_size = 128
        batch_size = 5
        in_channels = 1
        num_features = 64
        num_residual = 8
        unet = NoisyUNetGenerator2D(in_channels, num_features, num_residual)
        x = randn(FT, (img_size, img_size, in_channels, batch_size))
        noise = 2 .* rand(FT, (div(img_size, 4), div(img_size, 4), num_features * 4, batch_size)) .- 1
        y = unet(x, noise)
        @test unet(x, noise) |> size == (img_size, img_size, in_channels, batch_size)
        @test_throws AssertionError NoisyUNetGenerator2D(in_channels, num_features, 3)
    end
end
