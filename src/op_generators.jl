
using Flux
using Functors
using NeuralOperators

"""
    OperatorUNetGenerator
"""
struct OperatorUNetGenerator
    initial
    downblocks
    resblocks
    upblocks
    final
end

@functor OperatorUNetGenerator

function OperatorUNetGenerator(
    in_channels::Int,
    num_features::Int=8,
    num_residual::Int=4,
    device=gpu,
)
    initial_layer = Chain(
        OperatorConv(in_channels => num_features, (256, 256), FourierTransform, permuted=true),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    downsampling_blocks = [
        OperatorConvBlock(num_features, num_features * 2, (128, 128), true, device),
        OperatorConvBlock(num_features * 2, num_features * 4, (64, 64), true, device),
    ]

    resnet_blocks = Chain([OperatorResidualBlock(num_features * 4, (64, 64)) for _ in range(1, length=num_residual)]...)

    upsampling_blocks = [
        OperatorConvBlock(num_features * 4, num_features * 2, (128, 128), true, device),
        OperatorConvBlock(num_features * 2, num_features, (256, 256), true, device),
    ]
    final_layer = Chain(
        OperatorConv(num_features => in_channels, (256, 256), FourierTransform, permuted=true),
    )

    return OperatorUNetGenerator(
        initial_layer,
        downsampling_blocks,
        resnet_blocks,
        upsampling_blocks,
        final_layer
    ) |> device
end

function (net::OperatorUNetGenerator)(x)
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

"""
    OperatorConvBlock
"""
struct OperatorConvBlock
    conv
end

@functor OperatorConvBlock

function OperatorConvBlock(
    in_channels::Int,
    out_channels::Int,
    modes::Tuple,
    with_activation::Bool=true,
    device=gpu;
    kwargs...
)
    return ConvBlock(
        Chain(
            OperatorConv(in_channels => out_channels, modes, FourierTransform, permuted=true),
            InstanceNorm(out_channels),
            if with_activation
                x -> relu.(x)
            else
                identity
            end)
    ) |> device
end

function (net::OperatorConvBlock)(x)
    return net.conv(x)
end

"""
    OperatorResidualBlock
"""
struct OperatorResidualBlock
    block
end

@functor OperatorResidualBlock

function OperatorResidualBlock(
    in_channels::Int,
    modes::Tuple,
    device=gpu
)
    return OperatorResidualBlock(
        Chain(
            OperatorConvBlock(in_channels, in_channels, modes, true, device),
            OperatorConvBlock(in_channels, in_channels, modes, false, device)
        )
    ) |> device
end

function (net::OperatorResidualBlock)(x)
    return x + net.block(x)
end
