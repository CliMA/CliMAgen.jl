# fig
fig = Figure(resolution=(2000, 400))
n_boot = 10000
cil = 0.99
x = (1:256)/sqrt(2)
min_x = 2^0/sqrt(2)
max_x = 2^8/sqrt(2)
min_y = 1e-5
max_y = ch == 1 ? 1e0 : 1e1 

# low resolution data
spectra_lr = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== 0.0) .&& (spectra.channel .== ch), 1:256])]
real_l_lr, real_u_lr = get_spectra_bci(spectra_lr, n_boot, cil)
real_l_lr, real_u_lr = real_l_lr./(1:256).^2, real_u_lr./(1:256).^2

# mean
wn = 1.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,1], ylabel="Average power spectrum", title="k = $wn", xscale = log2, yscale = log10)
band!(x, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
if ch == 1
    lines!(x, (x).^(-5/3), color=:black, linestyle=:dash)
else
    lines!(x, 10*(x).^(-3), color=:black, linestyle=:dash)
end
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

wn = 2.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,2], title="k = $wn", xscale = log2, yscale = log10)
band!(x, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
if ch == 1
    lines!(x, (x).^(-5/3), color=:black, linestyle=:dash)
else
    lines!(x, 10*(x).^(-3), color=:black, linestyle=:dash)
end
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

wn = 4.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,3], xlabel="Wavenumber", title="k = $wn", xscale = log2, yscale = log10)
band!(x, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
if ch == 1
    lines!(x, (x).^(-5/3), color=:black, linestyle=:dash)
else
    lines!(x, 10*(x).^(-3), color=:black, linestyle=:dash)
end
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

wn = 8.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,4], title="k = $wn", xscale = log2, yscale = log10)
band!(x, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
if ch == 1
    lines!(x, (x).^(-5/3), color=:black, linestyle=:dash)
else
    lines!(x, 10*(x).^(-3), color=:black, linestyle=:dash)
end
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

wn = 16.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,5], title="k = $wn", xscale = log2, yscale = log10)
band!(x, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low resolution")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
if ch == 1
    lines!(x, (x).^(-5/3), color=:black, linestyle=:dash)
else
    lines!(x, 10*(x).^(-3), color=:black, linestyle=:dash)
end
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

axislegend(; position= :lb, titlesize= 22)

save("fig:spectra_ch$(ch).png", fig, px_per_unit = 2)