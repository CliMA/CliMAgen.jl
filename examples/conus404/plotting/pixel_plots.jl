function pixel_plots(xreal, xfake)
    fig = Figure(resolution=(800, 800), fontsize=24)
    n_boot = 100
    n_grid = 200
    cil = 0.99
    min_x, max_x = -2, 2 # need to be adjusted for our temperature range!
    real_l, real_u = get_pdf_bci(xreal[:], min_x, max_x, n_grid, n_boot, cil)
    fake_l, fake_u = get_pdf_bci(xfake[:], min_x, max_x, n_grid, n_boot, cil)
    ax = CairoMakie.Axis(fig[1,1], xlabel="Temperature", ylabel="Probability density", title="Test data")
    band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high res.")
    lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
    band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high res.")
    lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
    xlims!(ax, min_x, max_x)
    axislegend(; position= :lt, labelsize=16)
    save("fig:pixel_values_ch1.png", fig, px_per_unit = 2)
end