
using CUDA: CuArray
using Flux
using Functors

"""
    UNetGeneratorAR
"""
struct UNetGeneratorAR
    unet
end

@functor UNetGeneratorAR

function UNetGeneratorAR(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=9,
)
    @assert in_channels > 1
    unet = UNetGenerator(in_channels, num_features, num_residual)

    return UNetGeneratorAR(unet)
end

function (net_ar::UNetGeneratorAR)(x)
    FT = eltype(x)
    img_size_x, img_size_y, nchannels, nbatch = size(x)
    
    # TODO: not efficient but should work on GPU
    zero_field = CuArray(zeros(FT, (img_size_x, img_size_y, div(nchannels, 2), nbatch)))
    x1 = view(x, :, :, 1:div(nchannels, 2), :)
    
    y1 = view(net_ar.unet(cat(zero_field, x1, dims=3)), :, :, div(nchannels, 2)+1:nchannels, :)
    x2 = view(x, :, :, div(nchannels, 2)+1:nchannels, :)

    y = net_ar.unet(cat(y1, x2, dims=3))

    return y
end

"""
    UNetGenerator
"""
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
)
    initial_layer = Chain(
        Conv((7, 7), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    downsampling_blocks = [
        ConvBlock(3, num_features, num_features * 2, true, true; stride=2, pad=1),
        ConvBlock(3, num_features * 2, num_features * 4, true, true; stride=2, pad=1),
    ]

    resnet_blocks = Chain([ResidualBlock(num_features * 4) for _ in range(1, length=num_residual)]...)

    upsampling_blocks = [
        ConvBlock(3, num_features * 4, num_features * 2, true, false; stride=2, pad=SamePad()),
        ConvBlock(3, num_features * 2, num_features, true, false; stride=2, pad=SamePad()),
    ]

    final_layer = Chain(
        Conv((7, 7), num_features => in_channels; stride=1, pad=3)
    )

    return UNetGenerator(
        initial_layer,
        downsampling_blocks,
        resnet_blocks,
        upsampling_blocks,
        final_layer
    )
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

"""
    ConvBlock
"""
struct ConvBlock
    conv
end

@functor ConvBlock

function ConvBlock(
    kernel_size::Int,
    in_channels::Int,
    out_channels::Int,
    with_activation::Bool=true,
    down::Bool=true;
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
            if with_activation
                x -> relu.(x)
            else
                identity
            end)
    )
end

function (net::ConvBlock)(x)
    return net.conv(x)
end


"""
    ResidualBlock
"""
struct ResidualBlock
    block
end

@functor ResidualBlock

function ResidualBlock(
    in_channels::Int
)
    return ResidualBlock(
        Chain(
            ConvBlock(3, in_channels, in_channels, true, true; pad=1),
            ConvBlock(3, in_channels, in_channels, false, true; pad=1)
        )
    )
end

function (net::ResidualBlock)(x)
    return x + net.block(x)
end
