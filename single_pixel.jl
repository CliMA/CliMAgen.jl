# Time stepping scheme for a single step of length Δt
using Statistics
using Random
using Printf
using Plots

# Timestepper for variance exploding SDE
# should match Euler_Maruyama_sampling_ld
function Euler_Maruyama_step!(du,u,t,σ_min, σ_max, μ0, σ20, dt, cache; reverse::Bool = false, bias::Bool = false, k_bias = 0.0)
    # Deterministic step
    g = σ_min * (σ_max/σ_min)^t*sqrt(2*log(σ_max/σ_min)) # diffusion
    if reverse
        Dt = σ_min * (σ_max/σ_min)^t # marginal prob
        bias_drift = bias ? k_bias : 0.0 # from kA(x) = kx
        shift = bias ? Dt^2*bias_drift : 0.0
        score = @. -(u + shift - μ0)/(σ20 + Dt^2) + bias_drift
        @. du = g^2 * score # make positive since dt >0
    else
        du .= 0.0
    end
    u .+=  du .* dt
    # Stochastic step
    randn!(cache)
    du .= g * cache
    u .+=  sqrt(dt) .* du
end

# First create a data set of Gaussian random numbers
N = 100000
data = randn(N)
umin, umax = extrema(data)
dt = 0.025
@. data = (data - umin)/(umax-umin)*2 - 1
μ0 = mean(data)
σ20 = var(data)
σmin = 0.01
σmax = 2.0

# Now, test running forwards and backwards and make sure everything works as expected
# with the analytic score function.
# Try with different number of samples/initial conditions to
# account for statistical uncertainty
gen_means = []
gen_var = []
Narr = [10,100,1000,10000,100000]
for Nval in Narr
    u = data[1:Nval]
    cache = similar(u)
    du = similar(u)
    for t in 0.0:dt:1.0
        Euler_Maruyama_step!(du, u,t, σmin, σmax, μ0, σ20, dt, cache;)
    end
    # run in reverse
    for t in 1.0:-dt:0.0
        Euler_Maruyama_step!(du, u,t, σmin, σmax, μ0, σ20, dt, cache;reverse = true, bias = false)
    end
    push!(gen_means, mean(u))
    push!(gen_var, var(u))
end

plt1 = Plots.scatter(Narr, zeros(5) .+ μ0, label = "data")
Plots.scatter!(plt1, Narr, gen_means, yerr = sqrt.(gen_var) ./ sqrt.(Narr), label= "gen")
Plots.plot!(plt1, xlabel= "N samples", ylabel = "Mean",xaxis = :log10)

plt2 = Plots.scatter(Narr,  zeros(5) .+ σ20, label = "")
Plots.scatter!(plt2, Narr, gen_var, yerr = sqrt(2)*gen_var ./ sqrt.(Narr .-1), label= "") # Var[σ^2] = 2σ^4/(N-1)
Plots.plot!(plt2,  ylabel = "Variance", xaxis = :log10)
Plots.plot(plt2,plt1, layout =(2,1))
Plots.savefig("nobias.png")

# This seems to work - now try biasing
k_bias =0.0 # can also try 1.0 to see how it works
Z = mean(exp.(k_bias *data)) # expectation taken over data distribution
u = deepcopy(data)
cache = similar(u)
du = similar(u)
for t in 0.0:dt:1.0
    Euler_Maruyama_step!(du, u,t, σmin, σmax, μ0, σ20, dt, cache;)
end
# run in reverse
for t in 1.0:-dt:0.0
    Euler_Maruyama_step!(du, u,t, σmin, σmax, μ0, σ20, dt, cache;reverse = true, bias = true, k_bias = k_bias)
end

# The likelihood ratio should integrate to 1
lr =  @. Z*exp(-k_bias * u) # p_data/p_bias, where p_bias = p_data e^(kx)/Z
@printf "∫likelihood ratio q dx %.4f k_bias %.2lf\n" mean(lr) k_bias

# The standard deviation should not change, but the mean should
# but it doesnt do what I expect...
@printf "Means: Data %.4f Gen %.4f Expected %.4f\n" μ0 mean(u) μ0+σ20*k_bias
@printf "Means: Data - Gen %.4f Standard Error %.4f\n" (μ0 - mean(u)) sqrt(var(u)/N)
@printf "Variance: Data %.4f Gen %.4f Expected %.4f\n" σ20 var(u) σ20
@printf "Variance: Data - Gen %.4f Standard Error %.4f\n" (σ20 - var(u)) sqrt(2* var(u)/N)     # Var[σ^2] = 2σ^4/(N-1)
# histogram
Plots.histogram(data, label = "data")
Plots.histogram!(u, label = "gen biased", title = "k_bias = $(k_bias)")
Plots.savefig("histogram_$(k_bias).png")

# Copied directly from utils_analysis
function event_probability(a_m::Vector{FT},
                           lr::Vector{FT}
                           ) where {FT<:AbstractFloat}
    sort_indices = reverse(sortperm(a_m))
    a_sorted = a_m[sort_indices]
    lr_sorted = lr[sort_indices] 
    M = length(a_m)
    # γ = P(X > a)
    γ = cumsum(lr_sorted)./M
    # Compute uncertainty 
    γ² = cumsum(lr_sorted.^2.0)./M
    σ_γ = sqrt.(γ² .-  γ.^2.0)/sqrt(M)
    return a_sorted, γ, σ_γ
end
# event probability curve
em_data, p_data, σp_data = event_probability(data, ones(N))
Plots.plot(em_data, p_data, label = "data", xlabel = "value", ylabel="probability(x>value)")
em_gen, p_gen, σp_gen = event_probability(u, lr)
Plots.plot!(em_gen, p_gen, label = "gen", title = "k_bias = $(k_bias)", yaxis = :log)
Plots.savefig("event_probability_$(k_bias).png")



