function preprocessing(xtrain, preprocess_params_file; FT=Float32)
    #scale means and spatial variations separately
    x̄ = mean(xtrain, dims=(1, 2))
    maxtrain_mean = maximum(x̄, dims=4)
    mintrain_mean = minimum(x̄, dims=4)
    Δ̄ = maxtrain_mean .- mintrain_mean
    xp = xtrain .- x̄
    maxtrain_p = maximum(xp, dims=(1, 2, 4))
    mintrain_p = minimum(xp, dims=(1, 2, 4))
    Δp = maxtrain_p .- mintrain_p

    # To prevent dividing by zero
    Δ̄[Δ̄ .== 0] .= FT(1)
    Δp[Δp .== 0] .= FT(1)
    scaling = MeanSpatialScaling{FT}(mintrain_mean, Δ̄, mintrain_p, Δp)
    JLD2.save_object(preprocess_params_file, scaling)
end