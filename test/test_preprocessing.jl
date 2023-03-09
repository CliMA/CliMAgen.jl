@testset "Preprocessing" begin
    @testset "Scaling transforms" begin
        FT = Float32
        x = randn(FT,(64,64,3,100))
        scaling = StandardScaling{FT}(minimum(x, dims = (1,2,4)), maximum(x, dims = (1,2,4)))
        @test x ≈ invert_preprocessing(apply_preprocessing(x, scaling), scaling)

        x̄ = mean(x, dims=(1, 2))
        max_mean = maximum(x̄, dims=4)
        min_mean = minimum(x̄, dims=4)
        Δ̄ = max_mean .- min_mean
        xp = x .- x̄
        max_p = maximum(xp, dims=(1, 2, 4))
        min_p = minimum(xp, dims=(1, 2, 4))
        Δp = max_p .- min_p
        scaling = MeanSpatialScaling{FT}(min_mean, Δ̄, min_p, Δp)
        @test x ≈ invert_preprocessing(apply_preprocessing(x, scaling), scaling)

    end
end