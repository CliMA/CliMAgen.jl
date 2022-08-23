
using CUDA: CuArray
using Flux
using Functors


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
    num_residual::Int=9;
    out_channels::Int=in_channels,
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
    NoisyUNetGenerator

A UNetGenerator with an even number of resnet layers;
noise is added to the input after half of the resnet layers
operate.
"""
struct NoisyUNetGenerator
    initial
    downblocks
    first_resnet_block
    second_resnet_block
    upblocks
    final
end

@functor NoisyUNetGenerator

function NoisyUNetGenerator(
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
        ConvBlock(3, num_features, num_features * 2, true, true; stride=2, pad=1),
        ConvBlock(3, num_features * 2, num_features * 4, true, true; stride=2, pad=1),
    ]

    first_resnet_block = Chain([ResidualBlock(num_features * 4) for _ in range(1, length=resnet_block_length)]...)
    second_resnet_block = Chain([ResidualBlock(num_features * 4) for _ in range(1, length=resnet_block_length)]...)

    upsampling_blocks = [
        ConvBlock(3, num_features * 4, num_features * 2, true, false; stride=2, pad=SamePad()),
        ConvBlock(3, num_features * 2, num_features, true, false; stride=2, pad=SamePad()),
    ]

    final_layer = Chain(
        Conv((7, 7), num_features => in_channels; stride=1, pad=3)
    )

    return NoisyUNetGenerator(
        initial_layer,
        downsampling_blocks,
        first_resnet_block,
        second_resnet_block,
        upsampling_blocks,
        final_layer
    )
end

function (net::NoisyUNetGenerator)(x, r)
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
    PatchNet

    A wrapper structure that allows for patch-wise generation of 2D images.
    It uses another network like UNet as input.
    Assumes square inputs.
"""
struct PatchNet
    net
end

@functor PatchNet

function (patch::PatchNet)(x)
    nx, ny, nc, _ = size(x)

    # x and y patch index ranges
    # px1 = 1:div(nx, 2)
    # px2 = div(nx, 2)+1:nx
    # py1 = 1:div(ny, 2)
    # py2 = div(ny, 2)+1:ny
    # pc1 = 1:nc
    pc2 = nc+1:2nc

    # generate masks for each x_ij patch
    o = view(zero(x) .+ 1, 1:div(nx, 2), 1:div(ny, 2), :, :)
    z = view(zero(x), 1:div(nx, 2), 1:div(ny, 2), :, :)
    m11 = cat(cat(o, z, dims=1), cat(z, z, dims=1), dims=2)
    m12 = cat(cat(z, z, dims=1), cat(o, z, dims=1), dims=2)
    m21 = cat(cat(z, o, dims=1), cat(z, z, dims=1), dims=2)
    m22 = cat(cat(z, z, dims=1), cat(z, o, dims=1), dims=2)

    # generate y_ij recursively
    # y11
    y00 = cat(cat(z, z, dims=1), cat(z, z, dims=1), dims=2)
    input = cat(y00, x .* m11, dims=3)
    y11 = view(patch.net(input), :, :, pc2, :) .* m11
    # y12
    input = cat(y11, x .* m12, dims=3)
    y12 = view(patch.net(input), :, :, pc2, :) .* (m11 .+ m12)
    # y21
    input = cat(y12, x .* m21, dims=3)
    y21 = view(patch.net(input), :, :, pc2, :) .* (m11 .+ m12 .+ m21)
    # y22
    input = cat(y21, x .* m22, dims=3)
    y22 = view(patch.net(input), :, :, pc2, :)

    return m11 .* y11 .+ m12 .* y12 .+ m21 .* y21 .+ m22 .* y22
end

"""
    RecursiveNet

    A wrapper structure that allows for temporally consistent 2D images.
    It uses another network like UNet as input.
"""
struct RecursiveNet
    net_basic
    net_recursive
end

@functor RecursiveNet

function (rec::RecursiveNet)(x)
    nc = size(x, 3)

    # t1 and t2 time index ranges
    c1 = 1:div(nc, 2) # first time slice
    c2 = div(nc, 2)+1:nc # second time slice
    
    # incoming x patches sliced by time
    xt1 = view(x, :, :, c1, :)
    xt2 = view(x, :, :, c2, :)

    # generate yt1
    yt1 = rec.net_basic(xt1)

    # generate yt2
    input = cat(xt2, yt1, dims=3)
    yt2 = rec.net_recursive(input)

    # assemble full output
    y = cat(yt1, yt2, dims=3)

    return y
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
