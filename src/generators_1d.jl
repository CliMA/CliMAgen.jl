"""
    UNetGenerator1D
"""
struct UNetGenerator1D
    initial
    downblocks
    resblocks
    upblocks
    final
end

@functor UNetGenerator1D

function UNetGenerator1D(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=9
)
    initial_layer = Chain(
        Conv((7,), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    downsampling_blocks = [
        ConvBlock1D(3, num_features, num_features * 2, true, true; stride=2, pad=1),
        ConvBlock1D(3, num_features * 2, num_features * 4, true, true; stride=2, pad=1),
    ]

    resnet_blocks = Chain([ResidualBlock1D(num_features * 4) for _ in range(1, length=num_residual)]...)

    upsampling_blocks = [
        ConvBlock1D(3, num_features * 4, num_features * 2, true, false; stride=2, pad=SamePad()),
        ConvBlock1D(3, num_features * 2, num_features, true, false; stride=2, pad=SamePad()),
    ]

    final_layer = Chain(
        Conv((7,), num_features => in_channels; stride=1, pad=3)
    )

    return UNetGenerator1D(
        initial_layer,
        downsampling_blocks,
        resnet_blocks,
        upsampling_blocks,
        final_layer
    )
end

function (net::UNetGenerator1D)(x)
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
    NoisyUNetGenerator1D

A UNetGenerator1D with an even number of resnet layers;
noise is added to the input after half of the resnet layers
operate.
"""
struct NoisyUNetGenerator1D
    initial
    downblocks
    first_resnet_block
    second_resnet_block
    upblocks
    final
end

@functor NoisyUNetGenerator1D

function NoisyUNetGenerator1D(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=8,
)
    @assert iseven(num_residual)
    resnet_block_length = div(num_residual, 2)

    initial_layer = Chain(
        Conv((7,), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    downsampling_blocks = [
        ConvBlock1D(3, num_features, num_features * 2, true, true; stride=2, pad=1),
        ConvBlock1D(3, num_features * 2, num_features * 4, true, true; stride=2, pad=1),
    ]

    first_resnet_block = Chain([ResidualBlock1D(num_features * 4) for _ in range(1, length=resnet_block_length)]...)
    second_resnet_block = Chain([ResidualBlock1D(num_features * 4) for _ in range(1, length=resnet_block_length)]...)

    upsampling_blocks = [
        ConvBlock1D(3, num_features * 4, num_features * 2, true, false; stride=2, pad=SamePad()),
        ConvBlock1D(3, num_features * 2, num_features, true, false; stride=2, pad=SamePad()),
    ]

    final_layer = Chain(
        Conv((7,), num_features => in_channels; stride=1, pad=3)
    )

    return NoisyUNetGenerator1D(
        initial_layer,
        downsampling_blocks,
        first_resnet_block,
        second_resnet_block,
        upsampling_blocks,
        final_layer
    )
end

function (net::NoisyUNetGenerator1D)(x, r)
    input = net.initial(x)
    for layer in net.downblocks
        input = layer(input)
    end
    input = net.first_resnet_block(input)
    input = input .+ r # add random noise
    input = net.second_resnet_block(input)
    for layer in net.upblocks
        input = layer(input)
    end

    return tanh.(net.final(input))
end

"""
    ConvBlock1D
"""
struct ConvBlock1D
    conv
end

@functor ConvBlock1D

function ConvBlock1D(
    kernel_size::Int,
    in_channels::Int,
    out_channels::Int,
    with_activation::Bool=true,
    down::Bool=true;
    kwargs...
)
    return ConvBlock1D(
        Chain(
            if down
                Conv((kernel_size,), in_channels => out_channels; kwargs...)
            else
                ConvTranspose((kernel_size,), in_channels => out_channels; kwargs...)
            end,
            InstanceNorm(out_channels),
            if with_activation
                x -> relu.(x)
            else
                identity
            end)
    )
end

function (net::ConvBlock1D)(x)
    return net.conv(x)
end

"""
    ResidualBlock1D
"""
struct ResidualBlock1D
    block
end

@functor ResidualBlock1D

function ResidualBlock1D(
    in_channels::Int
)
    return ResidualBlock1D(
        Chain(
            ConvBlock1D(3, in_channels, in_channels, true, true; pad=1),
            ConvBlock1D(3, in_channels, in_channels, false, true; pad=1)
        )
    )
end

function (net::ResidualBlock1D)(x)
    return x + net.block(x)
end
