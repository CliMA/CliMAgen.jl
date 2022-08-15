
using CUDA: CuArray
using Flux
using Functors


"""
    PatchCNN2D
"""
struct PatchCNN2D
    net
end

@functor PatchCNN2D

function PatchUNet2D(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=9,
)
    net = UNetGenerator(3 * in_channels, num_features, num_residual)

    return PatchCNN2D(net)
end

function (cnn::PatchCNN2D)(x)
    img_size_x, img_size_y, nchannels, _ = size(x)
    img_size_x_half = div(img_size_x, 2)
    img_size_y_half = div(img_size_y, 2)

    # chunk pixel indices into 4 sections
    px1 = 1:img_size_x_half
    px2 = img_size_x_half+1:img_size_x
    py1 = 1:img_size_y_half
    py2 = img_size_y_half+1:img_size_y

    # chunk input into 4 patches
    x11 = view(x, px1, py1, :, :)
    x12 = view(x, px2, py1, :, :)
    x21 = view(x, px1, py2, :, :)
    x22 = view(x, px2, py2, :, :)

    # zero field for boundary input when the adjacent patch is missing
    zer = 0 .* similar(x11)

    # recursively call the network to patch things together
    y11 = view(cnn.net(cat(zer, zer, x11, dims=3)), :, :, (2*nchannels+1):3*nchannels, :)
    y12 = view(cnn.net(cat(y11, zer, x12, dims=3)), :, :, (2*nchannels+1):3*nchannels, :)
    y21 = view(cnn.net(cat(zer, y11, x21, dims=3)), :, :, (2*nchannels+1):3*nchannels, :)
    y22 = view(cnn.net(cat(y21, y12, x22, dims=3)), :, :, (2*nchannels+1):3*nchannels, :)

    # cat the output patches together
    y = cat(cat(y11, y12, dims=1), cat(y21, y22, dims=1), dims=2)

    return y
end

"""
    AutoregressiveCNN2D
"""
struct AutoregressiveCNN2D
    net
end

@functor AutoregressiveCNN2D

function AutoregressiveUNet2D(
    in_channels::Int,
    num_features::Int=64,
    num_residual::Int=9,
)
    net = UNetGenerator(in_channels, num_features, num_residual)

    return AutoregressiveCNN2D(net)
end

function (cnn::AutoregressiveCNN2D)(xt)
    nchannels = size(x)[3]
    nchannels_half = div(nchannels, 2)

    # time slices from channels
    # idea is that the first half of channels comes from the first time slice
    # and the second half of channels comes from the second time slice
    t1 = 1:nchannels_half
    t2 = nchannels_half+1:nchannels 

    # chunk input into 2 time slices
    xt1 = view(xt, :, :, t1, :)
    xt2 = view(xt, :, :, t2, :)

    # zero field for initial input when the previous time slice is missing
    zer = 0 .* similar(xt1)

    # recursively call the network to patch things together
    yt1 = view(cnn.net(cat(zer, xt1, dims=3)), :, :, t2, :)
    yt2 = view(cnn.net(cat(yt1, xt2, dims=3)), :, :, t2, :)

    # cat the output patches together
    y = cat(yt1, yt2, dims=3)

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
