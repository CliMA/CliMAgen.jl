using Downscaling
using Test

@testset "Downscaling.jl" begin
    using Random
    function test()
        img_channels = 3
        img_size = 180
        ## need to explicity type to avoid Slow fallback implementation 
        ## https://discourse.julialang.org/t/flux-con-warning/49456
        x = randn(Float32, (img_size, img_size, img_channels, 2))
        gen = Generator(img_channels, 9)
        println(size(gen(x)))
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
end
