using Flux
include("network.jl")

struct PatchDiscriminator
    initial
    model
end

@functor PatchDiscriminator
function PatchDiscriminator(
    in_channels::Int=3,
    features::Any=[64, 128, 256, 512],
    device = gpu
)
    layers = []
    channel = features[1]
    for index in range(2, length(features))
        if features[index] != last(features)
            push!(layers,
                Block(channel, features[index], stride=2, device=device)
            )
        else
            push!(layers,
                Block(channel, features[index], stride=1, device=device)
            )
        end
        channel = features[index]
    end
    push!(layers,
        Conv((4, 4), channel => 1; stride=1, pad=1)
    )
    return Discriminator(
        Chain(
            Conv((4, 4), in_channels => features[1]; stride=2, pad=1),
            x -> leakyrelu.(x, 0.2)
        ),
        Chain(layers...)
    ) |> device
end

function (net::PatchDiscriminator)(x)
    input = net.initial(x)
    return sigmoid.(net.model(input))
end


struct Block
    conv
end

@functor Block

function Block(
    in_channels::Int,
    out_channels::Int;
    stride::Int,
    device = gpu
)
    return Block(
        Chain(
            Conv((4, 4), in_channels => out_channels; stride=stride, pad=1),
            InstanceNorm(out_channels),
            x -> leakyrelu.(x, 0.2)
        )
    ) |> device
end

function (net::Block)(x)
    return net.conv(x)
end







using Random
function test()
    img_channels = 3
    img_size = 100
    ## need to explicity type to avoid Slow fallback implementation 
    ## https://discourse.julialang.org/t/flux-con-warning/49456
    x = randn(Float32, (img_size, img_size, img_channels, 5))
    preds = Discriminator()
    println(size(preds(x)))
end