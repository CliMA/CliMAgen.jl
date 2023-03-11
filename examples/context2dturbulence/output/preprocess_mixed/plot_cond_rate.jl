# fig
fig = Figure(resolution=(2000, 400))
n_boot = 10000
n_grid = 200
cil = 0.99
FT = Float64

# mean
min_x, max_x = 0, 30
wn = 1.0
real_l, real_u = get_pdf_bci(cond[(cond.isreal .== true) .&& (cond.wavenumber .== wn) , :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(cond[(cond.isreal .== false) .&& (cond.wavenumber .== wn), :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,1], ylabel="Probability distribution", title="k = $wn", yscale=log10)
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-6, 1e1)
#axislegend(; position= :lb, titlesize= 22)

wn = 2.0
real_l, real_u = get_pdf_bci(cond[(cond.isreal .== true) .&& (cond.wavenumber .== wn) , :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(cond[(cond.isreal .== false) .&& (cond.wavenumber .== wn), :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,2], title="k = $wn", yscale=log10)
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-6, 1e1)
#axislegend(; position= :lb, titlesize= 22)

wn = 4.0
real_l, real_u = get_pdf_bci(cond[(cond.isreal .== true) .&& (cond.wavenumber .== wn) , :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(cond[(cond.isreal .== false) .&& (cond.wavenumber .== wn), :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,3], xlabel="Condensation rate", title="k = $wn", yscale=log10)
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-6, 1e1)
#axislegend(; position= :lb, titlesize= 22)

wn = 8.0
real_l, real_u = get_pdf_bci(cond[(cond.isreal .== true) .&& (cond.wavenumber .== wn) , :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(cond[(cond.isreal .== false) .&& (cond.wavenumber .== wn), :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,4], title="k = $wn", yscale=log10)
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-6, 1e1)
#axislegend(; position= :lb, titlesize= 22)

wn = 16.0
real_l, real_u = get_pdf_bci(cond[(cond.isreal .== true) .&& (cond.wavenumber .== wn) , :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(cond[(cond.isreal .== false) .&& (cond.wavenumber .== wn), :].cond_rate, min_x, max_x, n_grid, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,5], title="k = $wn", yscale=log10)
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-6, 1e1)
axislegend(; position= :rt, titlesize= 22)

save("fig3:cond_rate.png", fig, px_per_unit = 2)