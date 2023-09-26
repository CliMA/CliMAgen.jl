Using Plots

function event_probability(a_m::Vector{FT},
    lr::Vector{FT}
    ) where {FT<:AbstractFloat}
    sort_indices = reverse(sortperm(a_m))
    a_sorted = a_m[sort_indices]
    lr_sorted = lr[sort_indices] 
    M = length(a_m)
    # γa = P(X > a)
    γ = cumsum(lr_sorted)./M

    # Compute uncertainty 
    γ² = cumsum(lr_sorted.^2.0)./M
    σ_γ = sqrt.(γ² .-  γ.^2.0)/sqrt(M)
    return a_sorted, γ, σ_γ
end
logscale =true
N = 10000
σ = 1
k = 1.0
Z = exp(k^2*σ^2/2)
x = randn(N)*σ

Ñ = 100
x̃ = randn(Ñ)*σ .+ k*σ^2
lr_gen = exp.(-k .* x̃)*Z
em, γ, σ_γ = event_probability(x̃, lr_gen);
Plots.plot(em, γ,  ribbon = (σ_γ, σ_γ), label = "IS, N = 1e2")

Ñ = 1000
x̃ = randn(Ñ)*σ .+ k*σ^2
lr_gen = exp.(-k .* x̃)*Z
em, γ, σ_γ = event_probability(x̃, lr_gen);
Plots.plot!(em, γ,  ribbon = (σ_γ, σ_γ), label = "IS, N = 1e3")

Ñ = N
x̃ = randn(Ñ)*σ .+ k*σ^2
lr_gen = exp.(-k .* x̃)*Z
em, γ, σ_γ = event_probability(x̃, lr_gen);
Plots.plot!(em, γ,  ribbon = (σ_γ, σ_γ), label = "IS, N = 1e4")

lr_train = ones(Float64, length(x));
em, γ, σ_γ = event_probability(x, lr_train);
Plots.plot!(em, γ,  ribbon = (σ_γ, σ_γ), label = "DS, N = 1e4", ylabel = "Probability", xlabel = "Event magnitude", margin = 20Plots.mm)
if logscale
    Plots.plot!(yaxis = :log, legend = :bottomleft)
    Plots.savefig("tmp_log.png")
else
    Plots.savefig("tmp_linear.png")
end
