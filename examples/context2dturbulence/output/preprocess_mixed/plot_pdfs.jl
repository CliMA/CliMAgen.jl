# fig
fig = Figure(resolution=(2000, 1600))
n_boot = 10000
n_grid = 200
cil = 0.99

# mean
min_x, max_x = -10, -1
wn = 1.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[1,1], ylabel="Density estimate", title="k = $wn")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[1,2], title="k = $wn")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[1,3], xlabel="Spatial mean", title="k = $wn")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[1,4], title="k = $wn")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[1,5], title="k = $wn")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
axislegend(; position= :lt, titlesize= 22)
xlims!(ax, min_x, max_x)

# variance
min_x, max_x = 0, 50
wn = 1.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[2,1], ylabel="Density estimate")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[2,2])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[2,3], xlabel="Spatial variance")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[2,4])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[2,5])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

# skewnewss
min_x, max_x = -300, 50
wn = 1.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[3,1], ylabel="Density estimate")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[3,2])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[3,3], xlabel="Spatial skewness")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[3,4])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[3,5])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

# kurtosis
min_x, max_x = -1000, 3000
wn = 1.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[4,1], ylabel="Density estimate")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[4,2])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[4,3], xlabel="Spatial kurtosis")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[4,4])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(fig[4,5])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)

save("fig1:pdfs.png", fig, px_per_unit = 2)