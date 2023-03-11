# fig
fig = Figure(resolution=(2000, 400))
n_boot = 10000
cil = 0.99

# mean
wn = 1.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,1], ylabel="Average power spectrum", title="k = $wn", xscale = log2, yscale = log10)
band!(1:256, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(1:256, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(1:256, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(1:256, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(1:256, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, (1:256).^(-5/3), color=:black, linestyle=:dash)
text!("k⁻⁵/³", position=(2^6,1.5e-3), space = :data, fontsize=18)
xlims!(ax, 2^0, 2^8)
ylims!(ax, 1e-5, 1e0)

wn = 2.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,2], title="k = $wn", xscale = log2, yscale = log10)
band!(1:256, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(1:256, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(1:256, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(1:256, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(1:256, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, (1:256).^(-5/3), color=:black, linestyle=:dash)
text!("k⁻⁵/³", position=(2^6,1.5e-3), space = :data, fontsize=18)
xlims!(ax, 2^0, 2^8)
ylims!(ax, 1e-5, 1e0)

wn = 4.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,3], xlabel="Wavenumber", title="k = $wn", xscale = log2, yscale = log10)
band!(1:256, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(1:256, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(1:256, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(1:256, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(1:256, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, (1:256).^(-5/3), color=:black, linestyle=:dash)
text!("k⁻⁵/³", position=(2^6,1.5e-3), space = :data, fontsize=18)
xlims!(ax, 2^0, 2^8)
ylims!(ax, 1e-5, 1e0)

wn = 8.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,4], title="k = $wn", xscale = log2, yscale = log10)
band!(1:256, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(1:256, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(1:256, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(1:256, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(1:256, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, (1:256).^(-5/3), color=:black, linestyle=:dash)
text!("k⁻⁵/³", position=(2^6,1.5e-3), space = :data, fontsize=18)
xlims!(ax, 2^0, 2^8)
ylims!(ax, 1e-5, 1e0)

wn = 16.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,5], title="k = $wn", xscale = log2, yscale = log10)
band!(1:256, real_l, real_u, color=(:orange, 0.3), label="real high resolution")
lines!(1:256, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(1:256, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(1:256, fake_l, fake_u, color=(:purple, 0.3), label="generated high resolution")
lines!(1:256, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
lines!(1:256, (1:256).^(-5/3), color=:black, linestyle=:dash)
text!("k⁻⁵/³", position=(2^6,1.5e-3), space = :data, fontsize=18)
xlims!(ax, 2^0, 2^8)
ylims!(ax, 1e-5, 1e0)

axislegend(; position= :lb, titlesize= 22)

save("fig2:spectra.png", fig, px_per_unit = 2)