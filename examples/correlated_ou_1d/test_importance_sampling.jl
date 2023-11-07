# This is a script that carries out a sanity check on
# our event_probability function
# with Gaussian data.

Using Plots
include("../utils_analysis.jl")# bring in event_probability function
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
