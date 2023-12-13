using LinearAlgebra
include("./rhs.jl")

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

function simulate(u, tspan, dt, dt_save, seed; model, progress = true)
    N² = size(u)[1]
    Random.seed!(abs(seed))
    deterministic_tendency! = make_deterministic_tendency(model)
    stochastic_increment! = make_stochastic_increment(model)
    du = similar(u)
    nsteps = Int(floor((tspan[2]-tspan[1])/dt))
    n_steps_per_save = Int(round(dt_save/dt))
    savesteps = 0:n_steps_per_save:nsteps - n_steps_per_save
    solution = zeros(FT, (N², Int(nsteps/n_steps_per_save)))
    solution[:, 1] .= u 
    if progress 
        indices = ProgressBar(1:nsteps)
    else
        indices = 1:nsteps
    end
    for i in indices
        t = tspan[1]+dt*(i-1)
        Euler_Maruyama_step!(du, u, t, deterministic_tendency!, stochastic_increment!, dt)
        if i ∈ savesteps
            save_index = Int(i/n_steps_per_save)
            solution[:, save_index+1] .= u
        end
    end
    return solution
end

function setup_correlation_matrix(N; FT = Float32)
    Γ = FT.(reshape(zeros(N^4), (N^2,N^2)))
    for i1 in 1:N
        for j1 in 1:N
            k1 = (j1-1)*N+i1
            for i2 in 1:N
                for j2 in 1:N
                    k2 = (j2-1)*N+i2
                    Γ[k1,k2] = FT(1/sqrt(min(abs(i1-i2),N-abs(i1-i2))^2 + min(abs(j1-j2),N-abs(j1-j2))^2+1))
                end
            end
        end
    end
    ΓL = cholesky(Γ).L
    W = zeros(FT, N*N)
    W_corr = similar(W)
    return ΓL, W, W_corr
end

function setup_ludo_model(σ, α, β, γ, N; FT = Float32)
    ΓL, W, W_corr = setup_correlation_matrix(N; FT = FT)
    return LudoDiffusionSDE(σ, α, β, γ, N, Periodic(), ΓL, W, W_corr)
end
