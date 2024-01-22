function distribution_plot(df, plotname, nsamples, kvalues, sigmavalues)
    fig = Figure(size=(800, 400), fontsize=24)
    ax = Axis(fig[1, 1], xlabel="Shift (σ) ", ylabel = "Event distribution")
    train_obs = df[df.train  .== 1.0, "observable"];
    μ = mean(train_obs)
    σ = std(train_obs)
    for i in 1:length(kvalues)
        k = kvalues[i]
        s = sigmavalues[i]
        obs = df[df.train  .== 0.0 .&& df.k .== k, "observable"]
        expected_obs = k == 0.0f0 ? train_obs : randn(nsamples) .*σ .+ (μ + k*σ^2)
        obs_label = k == 0.0f0 ? "Generated" : nothing
        expected_obs_label = k == 0.0f0 ? "Expected" : nothing
        violin!(ax, zeros(nsamples) .+ s, expected_obs, side = :left, color = :orange, show_median=true, label=expected_obs_label, width = 0.25)
        violin!(ax, zeros(nsamples) .+ s, obs, side = :right, color = :teal, show_median=true, label = obs_label, width = 0.25)
    end
    axislegend(ax; position= :lt, labelsize= 16)
    save(plotname, fig, px_per_unit = 2)
end