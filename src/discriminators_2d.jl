struct PatchDiscriminator2D
    initial
    model
end

@functor PatchDiscriminator2D

function PatchDiscriminator2D(
    in_channels::Int=3,
    features::Any=[64, 128, 256, 512]
)
    layers = []
    channel = features[1]
    for index in range(2, length(features))
        if features[index] != last(features)
            push!(layers,
                PatchBlock2D(channel, features[index], stride=2)
            )
        else
            push!(layers,
                PatchBlock2D(channel, features[index], stride=1)
            )
        end
        channel = features[index]
    end
    push!(layers,
        Conv((4, 4), channel => 1; stride=1, pad=1)
    )
    return PatchDiscriminator2D(
        Chain(
            Conv((4, 4), in_channels => features[1]; stride=2, pad=1),
            x -> leakyrelu.(x, 0.2f0)
        ),
        Chain(layers...)
    )
end

function (net::PatchDiscriminator2D)(x)
    input = net.initial(x)
    return sigmoid.(net.model(input))
end


struct PatchBlock2D
    conv
end

@functor PatchBlock2D

function PatchBlock2D(
    in_channels::Int,
    out_channels::Int;
    stride::Int
)
    return PatchBlock2D(
        Chain(
            Conv((4, 4), in_channels => out_channels; stride=stride, pad=1),
            InstanceNorm(out_channels),
            x -> leakyrelu.(x, 0.2f0)
        )
    )
end

function (net::PatchBlock2D)(x)
    return net.conv(x)
end
