struct ConvMixer
    tokenizer
    untokenizer
    mixer
    tembedder
end

function ConvMixer(inchannels::Int, dim::Int, depth::Int; ks=7, ps=2, σ=Flux.swish, embed_dim=256, scale=30.0f0)
    tembedder = Chain(
        GaussianFourierProjection(embed_dim, scale),
        Dense(embed_dim, embed_dim, σ),
    )

    tokenizer = Chain(
        Conv((ps, ps), inchannels => dim, σ; stride=ps),
    )

    mixer = []
    push!(mixer, (identity, Dense(embed_dim, dim)))
    for _ in 1:depth
        block = (
            Chain(
                GroupNorm(dim, 32),
                SkipConnection(
                    Chain(
                        Conv((ks, ks), dim => dim, σ; groups=dim, pad=SamePad()),
                        GroupNorm(dim, 32),
                    ),
                    +,
                ),
                Conv((1, 1), dim => dim, σ),
                GroupNorm(dim, 32),
            ),
            Dense(embed_dim, dim, σ),
        )
        push!(mixer, block)
    end

    untokenizer = Chain(
        ConvTranspose((ps, ps), dim => inchannels; stride=ps),
    )

    return ConvMixer(tokenizer, untokenizer, mixer, tembedder)
end

@functor ConvMixer

function (net::ConvMixer)(x, t)
    ht = net.tembedder(t)
    hx = net.tokenizer(x)

    # time-aware mixing
    for block in net.mixer
        spatial_mixer, time_conditioner = block
        hx = spatial_mixer(hx) .+ expand_dims(time_conditioner(ht), 2)
    end

    return net.untokenizer(hx) 
end
