# TODO!: Remove
using CliMADatasets
using KernelDensity

ds_hr = Turbulence2DContext(; split=:train, resolution=512, wavenumber=:all, fraction=0.2)[:][:, :, ch, :]
ds_hr = reshape(ds_hr, prod(size(ds_hr)))
ds_hr = ds_hr[1:100000]

ds_lr = Turbulence2DContext(; split=:train, resolution=64, wavenumber=:all)[:][:, :, ch, :]
ds_lr = reshape(ds_lr, prod(size(ds_lr)))
ds_lr = ds_lr[1:100000]

# fig
fig = Figure(resolution=(2000, 400))
n_boot = 10000
n_grid = 100
cil = 0.99
FT = Float64
if ch == 1
    min_x, max_x = -30, 3
    min_y, max_y = 1e-5, 1e0
else
    min_x, max_x = -20, 20
    min_y, max_y = 1e-2, 1e0
end
x = LinRange(min_x, max_x, 100)

# mean
wn = 1.0
real_l_hr, real_u_hr = get_pdf_bci(ds_hr, min_x, max_x, n_grid, n_boot, cil)
fake_l_hr, fake_u_hr = get_pdf_bci(ds_hr, min_x, max_x, n_grid, n_boot, cil)
real_l_lr, real_u_lr = get_pdf_bci(ds_lr, min_x, max_x, n_grid, n_boot, cil)
fake_l_lr, fake_u_lr = get_pdf_bci(ds_lr, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr, fake_l_lr, fake_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT), fake_l_lr .+ eps(FT), fake_u_lr .+ eps(FT)
ax = Axis(fig[1,1], ylabel="Probability distribution", title="k = $wn", yscale=log10)
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, fake_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)
#axislegend(; position= :lb, titlesize= 22)

wn = 2.0

real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,2], title="k = $wn", yscale=log10)
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, fake_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)
#axislegend(; position= :lb, titlesize= 22)

wn = 4.0

real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,3], xlabel="Condensation rate", title="k = $wn", yscale=log10)
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, fake_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)
#axislegend(; position= :lb, titlesize= 22)

wn = 8.0

real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,4], title="k = $wn", yscale=log10)
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, fake_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)
#axislegend(; position= :lb, titlesize= 22)

wn = 16.0

real_l, real_u, fake_l, fake_u = real_l .+ eps(FT), real_u .+ eps(FT), fake_l .+ eps(FT), fake_u .+ eps(FT)
ax = Axis(fig[1,5], title="k = $wn", yscale=log10)
band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, fake_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)
axislegend(; position= :rt, titlesize= 22)

save("fig:pdf_ch$(ch).png", fig, px_per_unit = 2)
