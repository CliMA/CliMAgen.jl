
using Flux
using Functors

struct UNetGenerator
    initial
    downblocks
    resblocks
    upblocks
    final
end

@functor UNetGenerator

function UNetGenerator(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=9,
    device=gpu,
)
    initial_layer = Chain(
        Conv((7, 7), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    downsampling_blocks = [
        ConvBlock(3, num_features, num_features * 2, true, true; stride=2, pad=1, device=device),
        ConvBlock(3, num_features * 2, num_features * 4, true, true; stride=2, pad=1, device=device),
    ]

    resnet_blocks = Chain([ResidualBlock(num_features * 4) for _ in range(1, length=num_residual)]...)

    upsampling_blocks = [
        ConvBlock(3, num_features * 4, num_features * 2, true, false; stride=2, pad=SamePad(), device=device),
        ConvBlock(3, num_features * 2, num_features, true, false; stride=2, pad=SamePad(), device=device),
    ]

    final_layer = Conv((7, 7), num_features => in_channels; stride=1, pad=3)

    return UNetGenerator(
        initial_layer, 
        downsampling_blocks, 
        resnet_blocks, 
        upsampling_blocks, 
        final_layer
    ) |> device
end

function (net::UNetGenerator)(x)
    input = net.initial(x)
    for layer in net.downblocks
        input = layer(input)
    end
    input = net.resblocks(input)
    for layer in net.upblocks
        input = layer(input)
    end
    return tanh.(net.final(input))
end

struct ConvBlock
    conv
end

@functor ConvBlock

function ConvBlock(
    kernel_size::Int,
    in_channels::Int,
    out_channels::Int,
    act::Bool=true,
    down::Bool=true,
    device = gpu;
    kwargs...
)
    return ConvBlock(
        Chain(
            if down
                Conv((kernel_size, kernel_size), in_channels => out_channels; kwargs...)
            else
                ConvTranspose((kernel_size, kernel_size), in_channels => out_channels; kwargs...)
            end,
            InstanceNorm(out_channels),
            if act
                x -> relu.(x)
            else
                identity
            end)
    ) |> device
end

function (net::ConvBlock)(x)
    return net.conv(x)
end

struct ResidualBlock
    block
end

@functor ResidualBlock

function ResidualBlock(
    in_channels::Int,
    device = gpu
)
    return ResidualBlock(
        Chain(
            ConvBlock(3, in_channels, in_channels, true, true; pad=1, device=device),
            ConvBlock(3, in_channels, in_channels, false, true; pad=1, device=device)
        )
    ) |> device
end

function (net::ResidualBlock)(x)
    return x + net.block(x)
end
