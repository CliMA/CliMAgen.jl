struct NoiseConditionalScoreNetworkVariantCNNBypass
    layers::NamedTuple
    mean_bypass_layers::NamedTuple
    time_layers::NamedTuple
end

"""
User Facing API for NoiseConditionalScoreNetwork architecture.
"""
function NoiseConditionalScoreNetworkVariantCNNBypass(; nspatial=2, num_residual=8, inchannels=1, channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    time_layers =  (
        gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
        linear=Dense(embed_dim, embed_dim, swish),
    )
    
    layers = (#Lifting
              conv1=Conv((3, 3), inchannels => channels[1], stride=1, pad=SamePad()),
              dense1=Dense(embed_dim, channels[1]),
              gnorm1=GroupNorm(channels[1], 4, swish),
              
              # Encoding
              conv2=Downsampling(channels[1] => channels[2], nspatial),
              dense2=Dense(embed_dim, channels[2]),
              gnorm2=GroupNorm(channels[2], 32, swish),
              
              conv3=Downsampling(channels[2] => channels[3], nspatial),
              dense3=Dense(embed_dim, channels[3]),
              gnorm3=GroupNorm(channels[3], 32, swish),
              
              conv4=Downsampling(channels[3] => channels[4], nspatial),
              dense4=Dense(embed_dim, channels[4]),
              
              # Residual Blocks
              resnet_blocks = 
              [ResnetBlockVariant(channels[end], nspatial, embed_dim, 0.0f0) for _ in range(1, length=num_residual)],
              
              # Decoding
              gnorm4=GroupNorm(channels[4], 32, swish),
              tconv4=Upsampling(channels[4] => channels[3], nspatial),
              denset4=Dense(embed_dim, channels[3]),
              tgnorm4=GroupNorm(channels[3], 32, swish),
              
              tconv3=Upsampling(channels[3]+channels[3] => channels[2], nspatial),
              denset3=Dense(embed_dim, channels[2]),
              tgnorm3=GroupNorm(channels[2], 32, swish),
              
              tconv2=Upsampling(channels[2]+channels[2] => channels[1], nspatial),
              denset2=Dense(embed_dim, channels[1]),
              tgnorm2=GroupNorm(channels[1], 32, swish),
              
              # Projection
              tconv1=Conv((3, 3), channels[1] + channels[1] => inchannels, stride=1, pad=SamePad()),
    )
    mean_bypass_layers = (# Lifting
                          conv1=Conv((3, 3), inchannels => channels[1], stride=1, pad=SamePad()),
                          dense1=Dense(embed_dim, channels[1]),
                          gnorm1=GroupNorm(channels[1], 4, swish),
                          
                          # Encoding
                          conv2=Downsampling(channels[1] => channels[2], nspatial),
                          dense2=Dense(embed_dim, channels[2]),
                          gnorm2=GroupNorm(channels[2], 32, swish),
                          
                          conv3=Downsampling(channels[2] => channels[3], nspatial),
                          dense3=Dense(embed_dim, channels[3]),
                          gnorm3=GroupNorm(channels[3], 32, swish),
                          
                          conv4=Downsampling(channels[3] => channels[4], nspatial),
                          dense4=Dense(embed_dim, channels[4]),
                          
                          # Residual Blocks
                          resnet_blocks = 
                              [ResnetBlockVariant(channels[end], nspatial, embed_dim, 0.0f0) for _ in range(1, length=num_residual)],
                          
                          # Decoding
                          gnorm4=GroupNorm(channels[4], 32, swish),
                          tconv4=Upsampling(channels[4] => channels[3], nspatial),
                          denset4=Dense(embed_dim, channels[3]),
                          tgnorm4=GroupNorm(channels[3], 32, swish),
                          
                          tconv3=Upsampling(channels[3]+channels[3] => channels[2], nspatial),
                          denset3=Dense(embed_dim, channels[2]),
                          tgnorm3=GroupNorm(channels[2], 32, swish),
                          
                          tconv2=Upsampling(channels[2]+channels[2] => channels[1], nspatial),
                          denset2=Dense(embed_dim, channels[1]),
                          tgnorm2=GroupNorm(channels[1], 32, swish),
                          
                          # Projection
                          tconv1=Conv((3, 3), channels[1] + channels[1] => inchannels, stride=1, pad=SamePad()),
                          )
    
    return NoiseConditionalScoreNetworkVariantCNNBypass(layers, mean_bypass_layers, time_layers)
end

@functor NoiseConditionalScoreNetworkVariantCNNBypass

function (net::NoiseConditionalScoreNetworkVariantCNNBypass)(x, t)
    # Embedding
    embed = net.time_layers.gaussfourierproj(t)
    embed = net.time_layers.linear(embed)

    # Encoder
    h1 = x .- mean(x, dims=(1,2)) # remove mean before input
    
    h1 = net.layers.conv1(h1)
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

    # middle
    h = h4
    for block in net.layers.resnet_blocks
        h = block(h, embed)
    end

    # Decoder
    h = net.layers.gnorm4(h)
    h = net.layers.tconv4(h)
    h = h .+ expand_dims(net.layers.denset4(embed), 2)
    h = net.layers.tgnorm4(h)
    h = net.layers.tconv3(cat(h, h3; dims=3))
    h = h .+ expand_dims(net.layers.denset3(embed), 2)
    h = net.layers.tgnorm3(h)
    h = net.layers.tconv2(cat(h, h2, dims=3))
    h = h .+ expand_dims(net.layers.denset2(embed), 2)
    h = net.layers.tgnorm2(h)
    h = net.layers.tconv1(cat(h, h1, dims=3))

    h = h .- mean(h, dims=(1,2)) # remove mean after output


    # Mean processing
    zero_img = similar(x) .* 0
    hm1 = mean(x, dims = (1,2)) .+ zero_img # extract mean
    hm1 = net.mean_bypass_layers.conv1(hm1)
    hm1 = hm1 .+ expand_dims(net.mean_bypass_layers.dense1(embed), 2)
    hm1 = net.mean_bypass_layers.gnorm1(hm1)
    hm2 = net.mean_bypass_layers.conv2(hm1)
    hm2 = hm2 .+ expand_dims(net.mean_bypass_layers.dense2(embed), 2)
    hm2 = net.mean_bypass_layers.gnorm2(hm2)
    hm3 = net.mean_bypass_layers.conv3(hm2)
    hm3 = hm3 .+ expand_dims(net.mean_bypass_layers.dense3(embed), 2)
    hm3 = net.mean_bypass_layers.gnorm3(hm3)
    hm4 = net.mean_bypass_layers.conv4(hm3)
    hm4 = hm4 .+ expand_dims(net.mean_bypass_layers.dense4(embed), 2)

    # middle
    hm = hm4
    for block in net.mean_bypass_layers.resnet_blocks
        hm = block(hm, embed)
    end

    # Decoder
    hm = net.mean_bypass_layers.gnorm4(hm)
    hm = net.mean_bypass_layers.tconv4(hm)
    hm = hm .+ expand_dims(net.mean_bypass_layers.denset4(embed), 2)
    hm = net.mean_bypass_layers.tgnorm4(hm)
    hm = net.mean_bypass_layers.tconv3(cat(hm, hm3; dims=3))
    hm = hm .+ expand_dims(net.mean_bypass_layers.denset3(embed), 2)
    hm = net.mean_bypass_layers.tgnorm3(hm)
    hm = net.mean_bypass_layers.tconv2(cat(hm, hm2, dims=3))
    hm = hm .+ expand_dims(net.mean_bypass_layers.denset2(embed), 2)
    hm = net.mean_bypass_layers.tgnorm2(hm)
    hm = net.mean_bypass_layers.tconv1(cat(hm, hm1, dims=3))

    hm =  mean(hm, dims = (1,2)) .+ zero_img # remove spatial variation

    return h .+ hm

end

"""
    ClimaGen.NoiseConditionalScoreNetwork\n

# Notes
Images stored in (spatial..., channels, batch) order. \n

# References
https://arxiv.org/abs/1505.04597
"""
struct NoiseConditionalScoreNetwork
    layers::NamedTuple
end

"""
User Facing API for NoiseConditionalScoreNetwork architecture.
"""
function NoiseConditionalScoreNetwork(; inchannels=1, channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    return NoiseConditionalScoreNetwork((
        gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
        linear=Dense(embed_dim, embed_dim, swish),

        # Lifting
        conv1=Conv((3, 3), inchannels => channels[1], stride=1, pad=SamePad(), bias=false),
        dense1=Dense(embed_dim, channels[1]),
        gnorm1=GroupNorm(channels[1], 4, swish),

        # Encoding
        conv2=Conv((3, 3), channels[1] => channels[2], stride=2, pad=SamePad(), bias=false),
        dense2=Dense(embed_dim, channels[2]),
        gnorm2=GroupNorm(channels[2], 32, swish),

        conv3=Conv((3, 3), channels[2] => channels[3], stride=2, pad=SamePad(), bias=false),
        dense3=Dense(embed_dim, channels[3]),
        gnorm3=GroupNorm(channels[3], 32, swish),

        conv4=Conv((3, 3), channels[3] => channels[4], stride=2, pad=SamePad(), bias=false),
        dense4=Dense(embed_dim, channels[4]),
        gnorm4=GroupNorm(channels[4], 32, swish),

        # Residual Blocks

        # Decoding
        tconv4=ConvTranspose((3, 3), channels[4] => channels[3], pad=SamePad(), stride=2, bias=false),
        denset4=Dense(embed_dim, channels[3]),
        tgnorm4=GroupNorm(channels[3], 32, swish),

        tconv3=ConvTranspose((3, 3), channels[3] + channels[3] => channels[2], pad=SamePad(), stride=2, bias=false),
        denset3=Dense(embed_dim, channels[2]),
        tgnorm3=GroupNorm(channels[2], 32, swish),

        tconv2=ConvTranspose((3, 3), channels[2] + channels[2] => channels[1], pad=SamePad(), stride=2, bias=false),
        denset2=Dense(embed_dim, channels[1]),
        tgnorm2=GroupNorm(channels[1], 32, swish),
        
        # Projection
        tconv1=ConvTranspose((3, 3), channels[1] + channels[1] => inchannels, stride=1, pad=SamePad()),
    ))
end

@functor NoiseConditionalScoreNetwork

function (net::NoiseConditionalScoreNetwork)(x, t)
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

    return h
end

struct NoiseConditionalScoreNetworkVariant
    layers::NamedTuple
    mean_bypass::Bool
    scale_mean_bypass::Bool
    shift_input::Bool
    shift_output::Bool
    gnorm::Bool
end

"""
User Facing API for NoiseConditionalScoreNetwork architecture.
"""
function NoiseConditionalScoreNetworkVariant(; mean_bypass = false, scale_mean_bypass=false, shift_input=false, shift_output=false, gnorm=false, nspatial=2, num_residual=8, inchannels=1, channels=[32, 64, 128, 256], embed_dim=256, scale=30.0f0)
    if scale_mean_bypass & !mean_bypass
        @error("Attempting to scale the mean bypass term without adding in a mean bypass connection.")
    end
    if gnorm & !mean_bypass
        @error("Attempting to gnorm without adding in a mean bypass connection.")
    end

    # Mean processing as indicated by boolean mean_bypass
    if mean_bypass
        if gnorm
            mean_bypass_layers = (
                mean_skip_1 = Conv((1, 1), inchannels => embed_dim),
                mean_skip_2 = Conv((1, 1), embed_dim => embed_dim),
                mean_skip_3 = Conv((1, 1), embed_dim => inchannels),
                mean_gnorm_1 = GroupNorm(embed_dim, 32, swish),
                mean_gnorm_2 = GroupNorm(embed_dim, 32, swish),
                mean_dense_1 = Dense(embed_dim, embed_dim),
                mean_dense_2 = Dense(embed_dim, embed_dim),
            )
        else
            mean_bypass_layers = (
                mean_skip_1 = Conv((1, 1), inchannels => embed_dim),
                mean_skip_2 = Conv((1, 1), embed_dim => embed_dim),
                mean_skip_3 = Conv((1, 1), embed_dim => inchannels),
                mean_dense_1 = Dense(embed_dim, embed_dim),
                mean_dense_2 = Dense(embed_dim, embed_dim),
            )
        end
    else
        mean_bypass_layers = ()
    end
    
    layers = (gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
              linear=Dense(embed_dim, embed_dim, swish),
              
              # Lifting
              conv1=Conv((3, 3), inchannels => channels[1], stride=1, pad=SamePad()),
              dense1=Dense(embed_dim, channels[1]),
              gnorm1=GroupNorm(channels[1], 4, swish),
              
              # Encoding
              conv2=Downsampling(channels[1] => channels[2], nspatial),
              dense2=Dense(embed_dim, channels[2]),
              gnorm2=GroupNorm(channels[2], 32, swish),
              
              conv3=Downsampling(channels[2] => channels[3], nspatial),
              dense3=Dense(embed_dim, channels[3]),
              gnorm3=GroupNorm(channels[3], 32, swish),
              
              conv4=Downsampling(channels[3] => channels[4], nspatial),
              dense4=Dense(embed_dim, channels[4]),
              
              # Residual Blocks
              resnet_blocks = 
              [ResnetBlockVariant(channels[end], nspatial, embed_dim, 0.0f0) for _ in range(1, length=num_residual)],
              
              # Decoding
              gnorm4=GroupNorm(channels[4], 32, swish),
              tconv4=Upsampling(channels[4] => channels[3], nspatial),
              denset4=Dense(embed_dim, channels[3]),
              tgnorm4=GroupNorm(channels[3], 32, swish),
              
              tconv3=Upsampling(channels[3]+channels[3] => channels[2], nspatial),
              denset3=Dense(embed_dim, channels[2]),
              tgnorm3=GroupNorm(channels[2], 32, swish),
              
              tconv2=Upsampling(channels[2]+channels[2] => channels[1], nspatial),
              denset2=Dense(embed_dim, channels[1]),
              tgnorm2=GroupNorm(channels[1], 32, swish),
              
              # Projection
              tconv1=Conv((3, 3), channels[1] + channels[1] => inchannels, stride=1, pad=SamePad()),
              mean_bypass_layers...
              )
    
    return NoiseConditionalScoreNetworkVariant(layers, mean_bypass, scale_mean_bypass, shift_input, shift_output, gnorm)
end

@functor NoiseConditionalScoreNetworkVariant

function (net::NoiseConditionalScoreNetworkVariant)(x, t)
    # Embedding
    embed = net.layers.gaussfourierproj(t)
    embed = net.layers.linear(embed)

    # Encoder
    if net.shift_input
        h1 = x .- mean(x, dims=(1,2)) # remove mean before input
    else
        h1 = x
    end
    
    h1 = net.layers.conv1(h1)
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

    # middle
    h = h4
    for block in net.layers.resnet_blocks
        h = block(h, embed)
    end

    # Decoder
    h = net.layers.gnorm4(h)
    h = net.layers.tconv4(h)
    h = h .+ expand_dims(net.layers.denset4(embed), 2)
    h = net.layers.tgnorm4(h)
    h = net.layers.tconv3(cat(h, h3; dims=3))
    h = h .+ expand_dims(net.layers.denset3(embed), 2)
    h = net.layers.tgnorm3(h)
    h = net.layers.tconv2(cat(h, h2, dims=3))
    h = h .+ expand_dims(net.layers.denset2(embed), 2)
    h = net.layers.tgnorm2(h)
    h = net.layers.tconv1(cat(h, h1, dims=3))
    if net.shift_output
        h = h .- mean(h, dims=(1,2)) # remove mean after output
    end

    # Mean processing
    if net.mean_bypass
        hm = net.layers.mean_skip_1(mean(x, dims=(1,2)))
        hm = hm .+ expand_dims(net.layers.mean_dense_1(embed), 2)
        if net.gnorm
            hm = net.layers.mean_gnorm_1(hm)
        end
        hm = net.layers.mean_skip_2(hm)
        hm = hm .+ expand_dims(net.layers.mean_dense_2(embed), 2)
        if net.gnorm
            hm = net.layers.mean_gnorm_2(hm)
        end
        hm = net.layers.mean_skip_3(hm)
        if net.scale_mean_bypass
            scale = convert(eltype(x), sqrt(prod(size(x)[1:ndims(x)-2])))
            hm = hm ./ scale
        end
        return h .+ hm
    else
        return h
    end
end

"""
Projection of Gaussian Noise onto a time vector

# Notes
This layer will help embed random times onto the frequency domain. \n
W is not trainable and is sampled once upon construction - see assertions below.

# References
https://arxiv.org/abs/2006.10739
"""
struct GaussianFourierProjection{FT}
    W::AbstractArray{FT}
end

function GaussianFourierProjection(embed_dim::Int, scale::FT) where {FT}
    W = randn(FT, embed_dim ÷ 2) .* scale
    return GaussianFourierProjection(W)
end

@functor GaussianFourierProjection

function (gfp::GaussianFourierProjection{FT})(t) where {FT}
    t_proj = t' .* gfp.W .* FT(2π)
    return [sin.(t_proj); cos.(t_proj)]
end

# layer is not trainable
Flux.params(::GaussianFourierProjection) = nothing

"""
    ClimaGen.DenoisingDiffusionNetwork\n

# Notes
Images stored in (spatial..., channels, batch) order. \n

# References
Ho, Jain, Abbeel: Denoising diffusion probabilistic models.
"""
struct DenoisingDiffusionNetwork
    layers::NamedTuple
end

"""
User Facing API for DenoisingDiffusionNetwork architecture.

N.B. this recreates a channels inflation of inchannels->nf->nf*1->nf*2->nf*2->nf*2, which is expressed
in the Song et al. code by inchannels, nf = 128, ch multiplier = [1,2,2,2].

Their embedding dimension is nf*4.

Here for now we set nf = 64, ch_multiplier = [1,2,2,4]...
"""
function DenoisingDiffusionNetwork(; kernel_size=3, nspatial=2, inchannels=1, channels=[64, 128, 128, 256], nembed=256, scale=30.0f0, p=0.1f0, σ=Flux.swish)
    conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
    pad = SamePad()
    
    return DenoisingDiffusionNetwork(
        (
        gaussfourierproj=GaussianFourierProjection(nembed, scale),
        embedding_network=Chain(Dense(nembed, nembed, σ), Dense(nembed, nembed)),

        lift = Conv(conv_kernel, inchannels => channels[1], pad=pad),

        # Encoding
        encode_block1a=ResnetBlock(channels[1] => channels[1], nspatial, nembed, p, σ),
        encode_block1b=ResnetBlock(channels[1] => channels[1], nspatial, nembed, p, σ),
        downsample1=Downsampling(channels[1]=>channels[1], nspatial),
            encode_block2a=ResnetBlock(channels[1] => channels[2], nspatial, nembed, p, σ),
            encode_block2b=ResnetBlock(channels[2] => channels[2], nspatial, nembed, p, σ),
            downsample2=Downsampling(channels[2]=> channels[2], nspatial),
                encode_block3a=ResnetBlock(channels[2] => channels[3], nspatial, nembed, p, σ),
                encode_block3b=ResnetBlock(channels[3] => channels[3], nspatial, nembed, p, σ),
                downsample3=Downsampling(channels[3]=>channels[3], nspatial),
                    encode_block4a=ResnetBlock(channels[3] => channels[4], nspatial, nembed, p, σ),
                    encode_block4b=ResnetBlock(channels[4] => channels[4], nspatial, nembed, p, σ),

                    # Transformations in middle
                    middle_transform1=ResnetBlock(channels[4] => channels[4], nspatial, nembed, p, σ),
                    middle_transform2=ResnetBlock(channels[4] => channels[4], nspatial, nembed, p, σ),

                    # Decoding
                    decode_block4a=ResnetBlock(channels[4] + channels[4] => channels[4], nspatial, nembed, p, σ),
                    decode_block4b=ResnetBlock(channels[4] + channels[4] => channels[4], nspatial, nembed, p, σ),
                    decode_block4c=ResnetBlock(channels[4] + channels[3] => channels[4], nspatial, nembed, p, σ),
                    upsample4=Upsampling(channels[4]=>channels[4], nspatial),
                decode_block3a=ResnetBlock(channels[4] + channels[3] => channels[3], nspatial, nembed, p, σ),
                decode_block3b=ResnetBlock(channels[3] + channels[3] => channels[3], nspatial, nembed, p, σ),
                decode_block3c=ResnetBlock(channels[3] + channels[2] => channels[3], nspatial, nembed, p, σ),
                upsample3=Upsampling(channels[3]=> channels[3], nspatial),
            decode_block2a=ResnetBlock(channels[3] + channels[2] => channels[2], nspatial, nembed, p, σ),
            decode_block2b=ResnetBlock(channels[2] + channels[2] => channels[2], nspatial, nembed, p, σ),
            decode_block2c=ResnetBlock(channels[2] + channels[1] => channels[2], nspatial, nembed, p, σ),
            upsample2=Upsampling(channels[2]=>channels[2], nspatial),
        decode_block1a=ResnetBlock(channels[2] + channels[1] => channels[1], nspatial, nembed, p, σ),
        decode_block1b=ResnetBlock(channels[1] + channels[1] => channels[1], nspatial, nembed, p, σ),
        decode_block1c=ResnetBlock(channels[1] + channels[1] => channels[1], nspatial, nembed, p, σ),

        # Transformations at end
        project=Chain(
            GroupNorm(channels[1], min(channels[1] ÷ 4, 32), σ),
            Conv(conv_kernel, channels[1] => inchannels, pad=pad)
        ),
    )
    )
end

@functor DenoisingDiffusionNetwork

function (net::DenoisingDiffusionNetwork)(x, t)
    # Embedding
    tembed = net.layers.gaussfourierproj(t)
    tembed = net.layers.embedding_network(tembed)

    # Lifting
    cx = net.layers.lift(x)

    # Encoder
    h1a = net.layers.encode_block1a(cx, tembed)
    h1b = net.layers.encode_block1b(h1a, tembed)
    h1c = net.layers.downsample1(h1b)

    h2a = net.layers.encode_block2a(h1c, tembed)
    h2b = net.layers.encode_block2b(h2a, tembed)
    h2c = net.layers.downsample2(h2b)

    h3a = net.layers.encode_block3a(h2c, tembed)
    h3b = net.layers.encode_block3b(h3a, tembed)
    h3c = net.layers.downsample3(h3b)

    h4a = net.layers.encode_block4a(h3c, tembed)
    h4b = net.layers.encode_block4b(h4a, tembed)

    # Middle transformations
    h = net.layers.middle_transform1(h4a, tembed)
    h = net.layers.middle_transform2(h, tembed)

    # Decoder
    h = net.layers.decode_block4a(cat(h, h4b; dims=3), tembed)
    h = net.layers.decode_block4b(cat(h, h4a; dims=3), tembed)
    h = net.layers.decode_block4c(cat(h, h3c; dims=3), tembed)
    h = net.layers.upsample4(h)

    h = net.layers.decode_block3a(cat(h, h3b; dims=3), tembed)
    h = net.layers.decode_block3b(cat(h, h3a; dims=3), tembed)
    h = net.layers.decode_block3c(cat(h, h2c; dims=3), tembed)
    h = net.layers.upsample3(h)

    h = net.layers.decode_block2a(cat(h, h2b; dims=3), tembed)
    h = net.layers.decode_block2b(cat(h, h2a; dims=3), tembed)
    h = net.layers.decode_block2c(cat(h, h1c; dims=3), tembed)
    h = net.layers.upsample2(h)

    h = net.layers.decode_block1a(cat(h, h1b; dims=3), tembed)
    h = net.layers.decode_block1b(cat(h, h1a; dims=3), tembed)
    h = net.layers.decode_block1c(cat(h, cx; dims=3), tembed)

    # End transformations
    return net.layers.project(h)
end

"""
    CliMAgen.ResnetBlock

ResNet block with GroupNorm and GaussianFourierProjection.

References:
https://arxiv.org/abs/1505.04597
https://arxiv.org/abs/1712.09763
"""
struct ResnetBlock
    norm1
    conv1
    norm2
    conv2
    dense
    dropout
    bypass
end

function ResnetBlock(channels::Pair, nspatial::Int, nembed::Int, p=0.1f0, σ=Flux.swish)
    # channels needs to be larger than 4
    @assert channels.first ÷ 4 > 0
    @assert channels.second ÷ 4 > 0

    # Require same input and output spatial size
    pad = SamePad()

    return ResnetBlock(
        GroupNorm(channels.first, min(channels.first ÷ 4, 32), σ),
        Conv(Tuple(3 for _ in 1:nspatial), channels, pad=pad),
        GroupNorm(channels.second, min(channels.second ÷ 4, 32), σ),
        Conv(Tuple(3 for _ in 1:nspatial), channels.second => channels.second, pad=pad),
        Dense(nembed => channels.second, σ),
        Dropout(p),
        Conv(Tuple(1 for _ in 1:nspatial), channels, pad=pad),
    )
end

@functor ResnetBlock

function (net::ResnetBlock)(x, tembed)
    # add on temporal embeddings to condition on time
    h = net.norm1(x)
    h = net.conv1(h) .+ expand_dims(net.dense(tembed), 2)

    # dropout is needed for low complexity datasets to
    # avoid overfitting
    h = net.norm2(h)
    h = net.dropout(h)
    h = net.conv2(h)

    return h .+ net.bypass(x)
end

"""
    CliMAgen.ResnetBlock

ResNet block with GroupNorm and GaussianFourierProjection.

References:
https://arxiv.org/abs/1505.04597
https://arxiv.org/abs/1712.09763
"""
struct ResnetBlockVariant
    norm1
    conv1
    norm2
    conv2
    dense
    dropout
end

function ResnetBlockVariant(channels::Int, nspatial::Int, nembed::Int, p=0.1f0, σ=Flux.swish)
    # channels needs to be larger than 4
    @assert channels ÷ 4 > 0

    # Require same input and output spatial size
    pad = SamePad()

    return ResnetBlockVariant(
        GroupNorm(channels, min(channels ÷ 4, 32), σ),
        Conv(Tuple(3 for _ in 1:nspatial), channels => channels, pad=pad),
        GroupNorm(channels, min(channels ÷ 4, 32), σ),
        Conv(Tuple(3 for _ in 1:nspatial), channels => channels, pad=pad),
        Dense(nembed => channels, σ),
        Dropout(p),
    )
end

@functor ResnetBlockVariant

function (net::ResnetBlockVariant)(x, tembed)
    # add on temporal embeddings to condition on time
    h = net.norm1(x)
    h = net.conv1(h) .+ expand_dims(net.dense(tembed), 2)

    # dropout is needed for low complexity datasets to
    # avoid overfitting
    h = net.norm2(h)
    h = net.dropout(h)
    h = net.conv2(h)

    return h .+ x
end

"""
    CliMAGen.AttentionBlock

Softmax attention block with group normalization and bypass connection.

References:
https://arxiv.org/abs/1712.09763
"""
struct AttentionBlock
    normalize
    queries
    keys
    values
    project
    bypass
end

function AttentionBlock(channels, nspatial, ngroups=32)
    @assert channels.first % ngroups == 0
    return AttentionBlock(
        GroupNorm(channels.first, ngroups),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
    )
end

@functor AttentionBlock

function (net::AttentionBlock)(x::AbstractArray{FT}) where {FT}
    nspatial = ndims(x) - 2
    nchannels = size(x, nspatial + 1)

    # use queries, keys, values framework to compute attention
    h = net.normalize(x)
    q = net.queries(h)
    k = net.keys(h)
    v = net.values(h)

    w = compute_attention_weights(k, q, Val(nspatial))

    # normalize since attention weights need to sum to one
    w = w ./ FT(sqrt(nchannels))
    w = Flux.softmax(w; dims=1:nspatial)

    h = apply_attention_weights(w, v, Val(nspatial))
    h = net.project(h)

    return h .+ net.bypass(x)
end

"""
    CliMAgen.compute_attention_weights(keys, queries, Val{nspatialdims})

Helper function that returns the softmax attention weights 
for given keys 'k' and queries 'q' by computing the dot product 
of the keys and queries across channels.
"""
compute_attention_weights(k, q, ::Val{2}) =
    @tullio w[h1, w1, h2, w2, b] := k[h1, w1, c, b] * q[h2, w2, c, b]

"""
    CliMAgen.apply_attention_weights(weights, values, Val{nspatialdims})

Helper function that applies attention weights for given weights 
'w' and values 'v' by taking the weighted sum of the values at 
each location and batch.
"""
apply_attention_weights(w, v, ::Val{2}) =
    @tullio v_weighted[h2, w2, c, b] := w[h1, w1, h2, w2, b] * v[h1, w1, c, b]


function Downsampling(channels::Pair, nspatial::Int, factor::Int=2, kernel_size::Int=3)
    conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
    return Conv(conv_kernel, channels, stride=factor, pad=SamePad())
end

"""
    CliMAgen.Upsampling(nspatial::Int, nchannels::Int, factor::Int=2, kernel_size::Int=3)

Checkerboard-save upsampling using nearest neighbor interpolation and convolution.

References:
https://distill.pub/2016/deconv-checkerboard/
"""
function Upsampling(channels::Pair{S,S}, nspatial::Int, factor::Int=2, kernel_size::Int=3) where {S}
    conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
    return Chain(
        Flux.Upsample(factor, :nearest),
        Conv(conv_kernel, channels, pad=SamePad())
    )
end
