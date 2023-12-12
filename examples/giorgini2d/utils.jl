using Plots
function convert_to_animation(x, time_stride, clims)
    init_frames = size(x)[3]
    x = x[:,:,1:time_stride:init_frames]
    frames = size(x)[3]
    animation = @animate for i = 1:frames
        heatmap(
            x[:,:,i],
            xaxis = false, yaxis = false, xticks = false, yticks = false,
            clims = clims
        )
    end
    return animation
end

function convert_to_animation1d(x, time_stride)
    init_frames = size(x)[2]
    x = x[:,1:time_stride:init_frames]
    frames = size(x)[2]
    animation = @animate for i = 1:frames
        plot(
            x[:,i],
            xaxis = false, yaxis = false, xticks = false, yticks = false,label = "", ylim = extrema(x)
        )
    end
    return animation
end

# Time stepping scheme for a single step of length Δt
function Euler_Maruyama_step!(du,u,t,f!,g!, dt)
    # Deterministic step
    du .= FT(0)
    f!(du,u,t)
    u .+=  du .* dt
    # Stochastic step
    du .= FT(0)
    g!(du,u,t)
    u .+=  sqrt(dt) .* du
    return u
end

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

