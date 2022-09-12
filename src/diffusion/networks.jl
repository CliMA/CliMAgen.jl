"""
    ClimaGen.NCNN\n

# Notes
Images stored in (spatial..., channels, batch) order. \n

# References
https://arxiv.org/abs/1505.04597
"""
struct NCNN
    layers::NamedTuple
end

"""
User Facing API for NCNN architecture.
"""
function NCNN(channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    return NCNN((
        gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
        linear=Dense(embed_dim, embed_dim, swish),

        # Encoding
        conv1=Conv((3, 3), 1 => channels[1], stride=1, bias=false),
        dense1=Dense(embed_dim, channels[1]),
        gnorm1=GroupNorm(channels[1], 4, swish), conv2=Conv((3, 3), channels[1] => channels[2], stride=2, bias=false),
        dense2=Dense(embed_dim, channels[2]),
        gnorm2=GroupNorm(channels[2], 32, swish), conv3=Conv((3, 3), channels[2] => channels[3], stride=2, bias=false),
        dense3=Dense(embed_dim, channels[3]),
        gnorm3=GroupNorm(channels[3], 32, swish), conv4=Conv((3, 3), channels[3] => channels[4], stride=2, bias=false),
        dense4=Dense(embed_dim, channels[4]),
        gnorm4=GroupNorm(channels[4], 32, swish),

        # Decoding
        tconv4=ConvTranspose((3, 3), channels[4] => channels[3], stride=2, bias=false),
        denset4=Dense(embed_dim, channels[3]),
        tgnorm4=GroupNorm(channels[3], 32, swish), tconv3=ConvTranspose((3, 3), channels[3] + channels[3] => channels[2], pad=(0, -1, 0, -1), stride=2, bias=false),
        denset3=Dense(embed_dim, channels[2]),
        tgnorm3=GroupNorm(channels[2], 32, swish), tconv2=ConvTranspose((3, 3), channels[2] + channels[2] => channels[1], pad=(0, -1, 0, -1), stride=2, bias=false),
        denset2=Dense(embed_dim, channels[1]),
        tgnorm2=GroupNorm(channels[1], 32, swish), tconv1=ConvTranspose((3, 3), channels[1] + channels[1] => 1, stride=1, bias=false),
    ))
end

@functor NCNN

function (net::NCNN)(x, t)
    # Embedding
    embed = net.layers.gaussfourierproj(t)
    embed = net.layers.linear(embed)

    # Encoder
    h1 = net.layers.conv1(x)
    h1 = h1 .+ expand_dims(net.layers.dense1(embed), 2)
    h1 = net.layers.gnorm1(h1)
    h2 = net.layers.conv2(h1)
    h2 = h2 .+ expand_dims(net.layers.dense2(embed), 2)
    h2 = net.layers.gnorm2(h2)
    h3 = net.layers.conv3(h2)
    h3 = h3 .+ expand_dims(net.layers.dense3(embed), 2)
    h3 = net.layers.gnorm3(h3)
    h4 = net.layers.conv4(h3)
    h4 = h4 .+ expand_dims(net.layers.dense4(embed), 2)
    h4 = net.layers.gnorm4(h4)

    # Decoder
    h = net.layers.tconv4(h4)
    h = h .+ expand_dims(net.layers.denset4(embed), 2)
    h = net.layers.tgnorm4(h)
    h = net.layers.tconv3(cat(h, h3; dims=3))
    h = h .+ expand_dims(net.layers.denset3(embed), 2)
    h = net.layers.tgnorm3(h)
    h = net.layers.tconv2(cat(h, h2, dims=3))
    h = h .+ expand_dims(net.layers.denset2(embed), 2)
    h = net.layers.tgnorm2(h)
    h = net.layers.tconv1(cat(h, h1, dims=3))

    # Scaling Factor
    return h
end

"""
Projection of Gaussian Noise onto a time vector

# Notes
This layer will help embed random times onto the frequency domain. \n
W is not trainable and is sampled once upon construction - see assertions below.

# References
https://arxiv.org/abs/2006.10739
"""
function GaussianFourierProjection(embed_dim, scale)
    # Instantiate W only once!
    W = randn(Float32, embed_dim ÷ 2) .* scale

    projection(t) = begin
        t_proj = t' .* W * Float32(2π)
        [sin.(t_proj); cos.(t_proj)]
    end

    return projection
end