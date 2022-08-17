
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
    net = UNetGenerator(4 * in_channels, num_features, num_residual)

    return PatchCNN2D(net)
end

function (cnn::PatchCNN2D)(x)
    img_size_i, img_size_j, nchannels, _ = size(x)

    # chunk pixel indices into 4 sections
    pi1 = 1:div(img_size_i, 2)
    pi2 = (div(img_size_i, 2)+1):img_size_i
    pj1 = 1:div(img_size_j, 2)
    pj2 = (div(img_size_j, 2)+1):img_size_j

    # chunk input into 4 patches
    x11 = view(x, pi1, pj1, :, :)
    x12 = view(x, pi1, pj2, :, :)
    x21 = view(x, pi2, pj1, :, :)
    x22 = view(x, pi2, pj2, :, :)

    # zero field for boundary input when the adjacent patch is missing
    zer = zero(x11)

    # recursively call the network to patch things together
    # y11 = view(cnn.net(cat(zer, zer, x11, dims=3)), :, :, (2*nchannels+1):3*nchannels, :)
    # y12 = view(cnn.net(cat(y11, zer, x12, dims=3)), :, :, (2*nchannels+1):3*nchannels, :)
    # y21 = view(cnn.net(cat(zer, y11, x21, dims=3)), :, :, (2*nchannels+1):3*nchannels, :)
    # y22 = view(cnn.net(cat(y21, y12, x22, dims=3)), :, :, (2*nchannels+1):3*nchannels, :)
    y11 = view(cnn.net(cat(zer, zer, zer, x11, dims=3)), :, :, (3*nchannels+1):4*nchannels, :)
    y12 = view(cnn.net(cat(zer, zer, zer, x12, dims=3)), :, :, (3*nchannels+1):4*nchannels, :)
    y21 = view(cnn.net(cat(zer, zer, zer, x21, dims=3)), :, :, (3*nchannels+1):4*nchannels, :)
    y22 = view(cnn.net(cat(zer, zer, zer, x22, dims=3)), :, :, (3*nchannels+1):4*nchannels, :)
    # y11 = view(cat(zer, zer, x11, dims=3), :, :, (2*nchannels+1):3*nchannels, :)
    # y12 = view(cat(zer, zer, x12, dims=3), :, :, (2*nchannels+1):3*nchannels, :)
    # y21 = view(cat(zer, zer, x21, dims=3), :, :, (2*nchannels+1):3*nchannels, :)
    # y22 = view(cat(zer, zer, x22, dims=3), :, :, (2*nchannels+1):3*nchannels, :)
    y_out = cnn.net(cat(y11, y12, y21, y22, dims=3))

    # chunk input into 4 patches
    y_out11 = view(y_out, :, :, (0*nchannels+1):1*nchannels, :)
    y_out12 = view(y_out, :, :, (1*nchannels+1):2*nchannels, :)
    y_out21 = view(y_out, :, :, (2*nchannels+1):3*nchannels, :)
    y_out22 = view(y_out, :, :, (3*nchannels+1):4*nchannels, :)

    # cat the output patches together
    y = cat(cat(y_out11, y_out21, dims=1), cat(y_out12, y_out22, dims=1), dims=2)

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
    nchannels = size(xt)[3]

    # time slices from channels
    # idea is that the first half of channels comes from the first time slice
    # and the second half of channels comes from the second time slice
    t1 = 1:div(nchannels, 2)
    t2 = (div(nchannels, 2)+1):nchannels

    # chunk input into 2 time slices
    xt1 = view(xt, :, :, t1, :)
    xt2 = view(xt, :, :, t2, :)

    # zero field for initial input when the previous time slice is missing
    zer = zero(xt1)

    # recursively call the network to patch things together
    yt1 = view(cnn.net(cat(zer, xt1, dims=3)), :, :, t2, :)
    y = cnn.net(cat(yt1, xt2, dims=3))

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

    return UNetGenerator(
        in_channels,
        in_channels,
        num_features,
        num_residual,
    )
end

function UNetGenerator(
    in_channels::Int,
    out_channels::Int,
    num_features::Int=64,
    num_residual::Int=9,
    num_down::Int=2,
    num_up::Int=2,
)
    initial_layer = Chain(
        Conv((7, 7), in_channels => num_features; stride=1, pad=3),
        InstanceNorm(num_features),
        x -> relu.(x)
    )

    # downsampling_blocks = [
    #     ConvBlock(3, num_features, num_features * 2, true, true; stride=2, pad=1),
    #     ConvBlock(3, num_features * 2, num_features * 4, true, true; stride=2, pad=1),
    # ]
    downsampling_blocks = Chain(
        [ConvBlock(3, num_features * 2^(i-1), num_features * 2^i, true, true; stride=2, pad=1) for i in 1:num_down]...    
    )

    resnet_blocks = Chain([ResidualBlock(num_features * 2^num_down) for _ in 1:length=num_residual]...)

    # upsampling_blocks = [
    #     ConvBlock(3, num_features * 4, num_features * 2, true, false; stride=2, pad=SamePad()),
    #     ConvBlock(3, num_features * 2, num_features, true, false; stride=2, pad=SamePad()),
    # ]
    upsampling_blocks = Chain(
        [ConvBlock(3, num_features * 2^i, num_features * 2^(i-1), true, false; stride=2, pad=SamePad()) for i in num_up:-1:1]...
    )

    final_layer = Chain(
        Conv((7, 7), num_features => out_channels; stride=1, pad=3)
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
    # for layer in net.downblocks
    #     input = layer(input)
    # end
    input = net.downblocks(input)
    input = net.resblocks(input)
    # for layer in net.upblocks
    #     input = layer(input)
    # end
    input = net.upblocks(input)
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
