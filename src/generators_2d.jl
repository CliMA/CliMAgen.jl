"""
    UNetGenerator2D
"""
struct UNetGenerator2D
    initial
    downblocks
    resblocks
    upblocks
    final
end

@functor UNetGenerator2D

function UNetGenerator2D(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=9,
)
    initial_layer = Chain(
        Conv((7, 7), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    downsampling_blocks = [
        ConvBlock2D(3, num_features, num_features * 2, true, true; stride=2, pad=1),
        ConvBlock2D(3, num_features * 2, num_features * 4, true, true; stride=2, pad=1),
    ]

    resnet_blocks = Chain([ResidualBlock2D(num_features * 4) for _ in range(1, length=num_residual)]...)

    upsampling_blocks = [
        ConvBlock2D(3, num_features * 4, num_features * 2, true, false; stride=2, pad=SamePad()),
        ConvBlock2D(3, num_features * 2, num_features, true, false; stride=2, pad=SamePad()),
    ]

    final_layer = Chain(
        Conv((7, 7), num_features => in_channels; stride=1, pad=3)
    )

    return UNetGenerator2D(
        initial_layer,
        downsampling_blocks,
        resnet_blocks,
        upsampling_blocks,
        final_layer
    )
end

function (net::UNetGenerator2D)(x)
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
    NoisyUNetGenerator2D

A UNetGenerator2D with an even number of resnet layers;
noise is added to the input after half of the resnet layers
operate.
"""
struct NoisyUNetGenerator2D
    initial
    downblocks
    first_resnet_block
    second_resnet_block
    upblocks
    final
end

@functor NoisyUNetGenerator2D

function NoisyUNetGenerator2D(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=8,
)
    @assert iseven(num_residual) 
    resnet_block_length = div(num_residual, 2)

    initial_layer = Chain(
        Conv((7, 7), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    downsampling_blocks = [
        ConvBlock2D(3, num_features, num_features * 2, true, true; stride=2, pad=1),
        ConvBlock2D(3, num_features * 2, num_features * 4, true, true; stride=2, pad=1),
    ]

    first_resnet_block = Chain([ResidualBlock2D(num_features * 4) for _ in range(1, length=resnet_block_length)]...)
    second_resnet_block = Chain([ResidualBlock2D(num_features * 4) for _ in range(1, length=resnet_block_length)]...)

    upsampling_blocks = [
        ConvBlock2D(3, num_features * 4, num_features * 2, true, false; stride=2, pad=SamePad()),
        ConvBlock2D(3, num_features * 2, num_features, true, false; stride=2, pad=SamePad()),
    ]

    final_layer = Chain(
        Conv((7, 7), num_features => in_channels; stride=1, pad=3)
    )

    return NoisyUNetGenerator2D(
        initial_layer,
        downsampling_blocks,
        first_resnet_block,
        second_resnet_block,
        upsampling_blocks,
        final_layer
    )
end

function (net::NoisyUNetGenerator2D)(x, r)
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
    ConvBlock2D
"""
struct ConvBlock2D
    conv
end

@functor ConvBlock2D

function ConvBlock2D(
    kernel_size::Int,
    in_channels::Int,
    out_channels::Int,
    with_activation::Bool=true,
    down::Bool=true;
    kwargs...
)
    return ConvBlock2D(
        Chain(
            if down
                Conv((kernel_size, kernel_size), in_channels => out_channels; kwargs...)
            else
                ConvTranspose((kernel_size, kernel_size), in_channels => out_channels; kwargs...)
            end,
            InstanceNorm(out_channels),
            if with_activation
                x -> relu.(x)
            else
                identity
            end)
    )
end

function (net::ConvBlock2D)(x)
    return net.conv(x)
end


"""
    ResidualBlock2D
"""
struct ResidualBlock2D
    block
end

@functor ResidualBlock2D

function ResidualBlock2D(
    in_channels::Int
)
    return ResidualBlock2D(
        Chain(
            ConvBlock2D(3, in_channels, in_channels, true, true; pad=1),
            ConvBlock2D(3, in_channels, in_channels, false, true; pad=1)
        )
    )
end

function (net::ResidualBlock2D)(x)
    return x + net.block(x)
end
