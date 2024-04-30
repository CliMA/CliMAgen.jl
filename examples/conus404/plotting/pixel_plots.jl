function pixel_plots(x1, x2, labels, plotname)
    fig = Figure(resolution=(800, 800), fontsize=24)
    n_boot = 100
    n_grid = 200
    cil = 0.99
    min_x, max_x = 240, 320 # need to be adjusted for our temperature range!
    x1_l, x1_u = get_pdf_bci(x1[:], min_x, max_x, n_grid, n_boot, cil)
    x2_l, x2_u = get_pdf_bci(x2[:], min_x, max_x, n_grid, n_boot, cil)
    ax = CairoMakie.Axis(fig[1,1], xlabel="Temperature", ylabel="Probability density")
    band!(LinRange(min_x, max_x, n_grid), x1_l, x1_u, color=(:orange, 0.3), label=labels[1])
    lines!(LinRange(min_x, max_x, n_grid), x1_l, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(LinRange(min_x, max_x, n_grid), x1_u, color=(:orange, 0.5), strokewidth = 1.5)
    band!(LinRange(min_x, max_x, n_grid), x2_l, x2_u, color=(:purple, 0.3), label=labels[2])
    lines!(LinRange(min_x, max_x, n_grid), x2_l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(LinRange(min_x, max_x, n_grid), x2_u, color=(:purple, 0.5), strokewidth = 1.5)
    xlims!(ax, min_x, max_x)
    axislegend(; position= :lt, labelsize=16)
    save(plotname, fig, px_per_unit = 2)
end

function simple_histogram_plot(x1, x2, labels, plotname)
    fig = Figure(resolution=(800, 800), fontsize=24)
    ax = CairoMakie.Axis(fig[1,1], xlabel="Temperature", ylabel="Count")
    hist!(x1[:], color=(:orange, 0.5), label=labels[1],normalization=:pdf, bins = 60)
    hist!(x2[:], color=(:purple, 0.5), label=labels[2], normalization=:pdf, bins = 60)
    xlims!(ax, extrema(x1)...)
    axislegend(; position= :lt, labelsize=16)
    save(plotname, fig, px_per_unit = 2)
end