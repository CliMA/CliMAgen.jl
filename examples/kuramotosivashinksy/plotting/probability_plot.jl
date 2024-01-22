function probability_plot(df, plotname, nsamples, k, sigma)
    fig = Figure(size=(600, 600), fontsize=24)
    train = df[df.train  .== 1.0, "observable"];
    train_lr = Float32.(df[df.train  .== 1.0, "likelihood_ratio"];)
    gen = df[df.train  .== 0.0 .&& df.k .== k, "observable"]
    gen_lr = Float32.(df[df.train  .== 0.0 .&& df.k .== k, "likelihood_ratio"])

    truth_event, truth_p, truth_sp = event_probability(train, train_lr)
    train_event, train_p, train_sp = event_probability(train[1:nsamples], train_lr[1:nsamples])
    gen_event, gen_p, gen_sp = event_probability(gen, gen_lr)
    ax = Axis(fig[1, 1], xlabel="Event Magnitude", ylabel = "Exceedance Probability",yscale=log10)
    band!(truth_event, truth_p .- truth_sp, truth_p .+ truth_sp, color=(:orange, 0.3), label="Data, N=4000")
    lines!(truth_event, truth_p .- truth_sp, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(truth_event, truth_p .+ truth_sp, color=(:orange, 0.5), strokewidth = 1.5)
    band!(train_event, train_p .- train_sp, train_p .+ train_sp, color=(:purple, 0.3), label="Data, N=$nsamples")
    lines!(train_event, train_p .- train_sp, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(train_event, train_p .+ train_sp, color=(:purple, 0.5), strokewidth = 1.5)
    band!(gen_event, gen_p .- gen_sp, gen_p .+ gen_sp, color=(:teal, 0.1), label="Generated, N=$nsamples")
    lines!(gen_event, gen_p .- gen_sp, color=(:teal, 0.2), strokewidth = 1.5)
    lines!(gen_event, gen_p .+ gen_sp, color=(:teal, 0.2), strokewidth = 1.5)
    axislegend(ax; position= :lb, labelsize= 16)
    save(plotname, fig, px_per_unit = 2)
end

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