"""
 2D Model
"""
struct LinearDiffusionSDE{FT <:AbstractFloat, CM, BC}
    σ::FT
    θ::FT
    N::Int
    bc::BC
    ΓL::CM
    W_cache::Vector{FT}
    W_corr::Vector{FT}
end

"""
    LudoDiffusionSDE

du = [(1/2 ∇² - β) tanh(γu) - α] dt + σdW
"""
struct LudoDiffusionSDE{FT <:AbstractFloat, CM, BC}
    σ::FT
    α::FT
    β::FT
    γ::FT
    N::Int
    bc::BC
    ΓL::CM
    W_cache::Vector{FT}
    W_corr::Vector{FT}
end

"""
 1D Model
"""
struct LinearDiffusion1dSDE{FT <:AbstractFloat, CM, BC}
    σ::FT
    θ::FT
    N::Int
    bc::BC
    ΓL::CM
    W_cache::Vector{FT}
    W_corr::Vector{FT}
end

abstract type AbstractBoundaryConditions end

struct Periodic <: AbstractBoundaryConditions end

struct Dirichlet{FT} <: AbstractBoundaryConditions
    boundary_value::FT
end

function make_deterministic_tendency(model::LinearDiffusionSDE{FT, CM, Dirichlet{FT}}) where {FT, CM}
    function deterministic_tendency!(du,u,t)
        N = model.N
        θ = model.θ
        u_bc = model.bc.boundary_value
        du .= FT(0)
        for i in 1:N
            for j in 1:N
                k = (j-1)*N+i
                k_ip1 = k+1
                k_im1 = k-1
                k_jp1 = k + N
                k_jm1 = k - N
                # At the boundary, we have u_bc on the face
                # while u[k] is at the cell center. -> Δx -> Δx/2
                if i == N          
                    du[k] += θ*((u[k_im1] - u[k]) - (u[k] - u_bc)/2)
                elseif i ==1
                    du[k] += θ*((u_bc - u[k])/2 - (u[k] - u[k_ip1]))
                else
                    du[k] += θ*((u[k_im1] - u[k]) - (u[k] - u[k_ip1]))
                end
                if j == N
                    du[k] += θ*((u_bc - u[k])/2 - (u[k] - u[k_jm1]))
                elseif j ==1
                    du[k] += θ*((u[k_jp1] - u[k]) - (u[k] - u_bc)/2)
                else
                    du[k] += θ*((u[k_jp1] - u[k]) - (u[k] - u[k_jm1]))
                end
            end
        end
    end
    return deterministic_tendency!
end

function make_deterministic_tendency(model::LudoDiffusionSDE{FT, CM, Periodic}) where {FT, CM}
    function deterministic_tendency!(du,u,t)
        N = model.N
        α = model.α
        β = model.β
        γ = model.γ

        # initialize tendency
        du .= FT(0)
        
        # temporary nonlinear vector
        # In the future, we could pre-allocate as needed
        v = @. tanh(γ*u)

        # apply Laplacian to nonlinear vector
        for i in 1:N
            for j in 1:N
                k = (j-1)*N+i
                k_ip1 = k+1
                k_im1 = k-1
                k_jp1 = k + N
                k_jm1 = k - N
                # Periodic BC: Δ in denominator of ∇v is the same at the boundary
                # or in the interiori.
                # Factor of 2 is definitional
                if i == N
                    v_bc = v[(j-1)*N+1]
                    du[k] += ((v[k_im1] - v[k]) - (v[k] - v_bc))/2
                elseif i == 1
                    v_bc = v[(j-1)*N+N]
                    du[k] += ((v_bc - v[k])/2 - (v[k] - v[k_ip1]))/2
                else
                    du[k] += ((v[k_im1] - v[k]) - (v[k] - v[k_ip1]))/2
                end
                if j == N
                    v_bc = v[i]
                    du[k] += ((v_bc - v[k])/2 - (v[k] - v[k_jm1]))/2
                elseif j == 1
                    v_bc = v[(N-1)*N+i]
                    du[k] += ((v[k_jp1] - v[k]) - (v[k] - v_bc))/2
                else
                    du[k] += ((v[k_jp1] - v[k]) - (v[k] - v[k_jm1]))/2
                end
            end
        end
 
        # the rest of Ludo's model
        @. du += -β * v - α
    end
    return deterministic_tendency!
end

function make_deterministic_tendency(model::LinearDiffusion1dSDE{FT, CM, Dirichlet{FT}}) where {FT, CM}
    function deterministic_tendency!(du,u,t)
        N = model.N
        θ = model.θ
        u_bc = model.bc.boundary_value
        du .= FT(0)
        for k in 1:N
            # At the boundary, we have u_bc on the face
            # while u[k] is at the cell center. -> Δx -> Δx/2
            if k == 1
                du[k] += θ*((u[k+1] - u[k]) - (u[k] - u_bc)/2)
            elseif k == N
                du[k] += θ*((u_bc - u[k])/2 - (u[k] - u[k-1]))
            else
                du[k] += θ*((u[k+1] - u[k]) - (u[k] - u[k - 1]))
            end
        end
    end
    return deterministic_tendency!
end

function make_stochastic_increment(model::Union{LudoDiffusionSDE{FT}, LinearDiffusionSDE{FT}, LinearDiffusion1dSDE{FT}}) where {FT}
    function stochastic_increment!(du,u,t)
        ΓL = model.ΓL
        W_cache = model.W_cache
        W_corr = model.W_corr
        σ = model.σ
        du.= FT(0)
        randn!(W_cache)
        mul!(W_corr,ΓL, W_cache)
        du .= σ .* W_corr
    end
    return stochastic_increment!
end
