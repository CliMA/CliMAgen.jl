# Time stepping scheme for a single step of length Δt
function Euler_Maruyama_step!(du,u,t,σ_min, σ_max, μ0, σ20, dt, cache; reverse::Bool = false, bias::Bool = false)
    # Deterministic step
    du .= 0
    g = σ_min * (σ_max/σ_min)^t*sqrt(2*log(σ_max/σ_min))
    if reverse
        Dt = σ_min * (σ_max/σ_min)^t
        bias_drift = bias ? 1.0 : 0.0 # equivalent to A(x) = x and k = 1
        shift = bias ? Dt^2*bias_drift : 0.0
        @. du = -g^2 * (u + shift - μ0)/(σ20 + Dt^2) + bias_drift
    end
    u .+=  du .* dt
    
    # Stochastic step
    du .= 0.0
    randn!(cache)
    du .= g * cache
    u .+=  sqrt(dt) .* du
end

# First create a data set of Gaussian random numbers
upool = randn(100000)
umin, umax = extrema(upool)
dt = 0.025
@. upool = (upool - umin)/(umax-umin)*2 - 1
μ0 = mean(upool)
σ20 = var(upool)
σmin = 0.01
σmax = 2.0

# Now, test running forwards and backwards and make sure everything works as expected
# with the analytic score function.
# Try with different number of samples/initial conditions to
# account for statistical uncertainty
gen_means = []
gen_var = []

for N in Narr
    u = upool[1:N]
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
Plots.scatter!(plt1, Narr, gen_means, yerr = gen_stds ./ sqrt.(Narr), label= "gen")
Plots.plot!(plt1, xlabel= "N samples", ylabel = "Mean",xaxis = :log10)

plt2 = Plots.scatter(Narr,  zeros(5) .+ σ20, label = "")
Plots.scatter!(plt2, Narr, gen_var, yerr = sqrt(2)*gen_var ./ sqrt.(Narr .-1), label= "") # Var[σ^2] = 2σ^4/(N-1)
Plots.plot!(plt2,  ylabel = "Variance", xaxis = :log10)
Plots.plot(plt2,plt1, layout =(2,1))


# This seems to work - now try biasing
gen_means = []
gen_var = []
for N in Narr
    u = upool[1:N]
    cache = similar(u)
    du = similar(u)
    for t in 0.0:dt:1.0
        Euler_Maruyama_step!(du, u,t, σmin, σmax, μ0, σ20, dt, cache;)
    end
    # run in reverse
    for t in 1.0:-dt:0.0
        Euler_Maruyama_step!(du, u,t, σmin, σmax, μ0, σ20, dt, cache;reverse = true, bias = true)
    end
    push!(gen_means, mean(u))
    push!(gen_var, var(u))
end

plt1 = Plots.scatter(Narr, zeros(5) .+ μ0, label = "data")
Plots.scatter!(plt1, Narr, gen_means, yerr = gen_stds ./ sqrt.(Narr), label= "gen")
expected = μ0 +σ20 # from completing the square with k = 1 and A(x) = x
Plots.scatter!(plt1,Narr, zeros(5) .+ expected, label= "Expected biased mean")
Plots.plot!(plt1, xlabel= "N samples", ylabel = "Mean",xaxis = :log10)

plt2 = Plots.scatter(Narr,  zeros(5) .+ σ20, label = "")
Plots.scatter!(plt2, Narr, gen_var, yerr = sqrt(2)*gen_var ./ sqrt.(Narr .-1), label= "") # Var[σ^2] = 2σ^4/(N-1)
Plots.plot!(plt2,  ylabel = "Variance", xaxis = :log10)
Plots.plot(plt2,plt1, layout =(2,1))
