using Downscaling
using Flux
using Test
using Random

@testset "Downscaling.jl" begin
    # using Random
    # function test()
    #     img_channels = 3
    #     img_size = 180
    #     ## need to explicity type to avoid Slow fallback implementation 
    #     ## https://discourse.julialang.org/t/flux-con-warning/49456
    #     x = randn(Float32, (img_size, img_size, img_channels, 2))
    #     gen = Generator(img_channels, 9)
    #     println(size(gen(x)))
    # end

    # using Random
    # function test()
    #     img_channels = 3
    #     img_size = 100
    #     ## need to explicity type to avoid Slow fallback implementation 
    #     ## https://discourse.julialang.org/t/flux-con-warning/49456
    #     x = randn(Float32, (img_size, img_size, img_channels, 5))
    #     preds = Discriminator()
    #     println(size(preds(x)))
    # end

    # PatchBlock
    in_channels = 3
    out_channels = 7
    img_size = 97
    batch_size = 13
    stride = 2
    patch_block = PatchBlock(in_channels, out_channels, stride=stride)
    x = randn(Float32, (img_size, img_size, in_channels, batch_size))
    @test patch_block(x) |> size == (48, 48, 7, 13)

    # PatchDiscriminator
    in_channels = 3
    out_channels = 7
    img_size = 97
    batch_size = 13
    stride = 2
    patch_block = PatchDiscriminator(in_channels)
    x = randn(Float32, (img_size, img_size, in_channels, batch_size))
    @test patch_block(x) |> size == (10, 10, 1, 13)
end
