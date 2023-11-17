# fig
fig = Figure(resolution=(400, 400))
n_boot = 10000
cil = 0.99
x = (1:256)/sqrt(2)
min_x = 2^0/sqrt(2)
max_x = 2^8/sqrt(2)
min_y = 1e-5
max_y = ch == 1 ? 1e0 : 1e1 

# mean
wn = 0.0
spectra_real = [Array(r) for r in eachrow(spectra[(spectra.isreal .== true) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
spectra_false = [Array(r) for r in eachrow(spectra[(spectra.isreal .== false) .&& (spectra.wavenumber .== wn) .&& (spectra.channel .== ch), 1:256])]
real_l, real_u = get_spectra_bci(spectra_real, n_boot, cil)
fake_l, fake_u = get_spectra_bci(spectra_false, n_boot, cil)
real_l, real_u, fake_l, fake_u = real_l./(1:256).^2, real_u./(1:256).^2, fake_l./(1:256).^2, fake_u./(1:256).^2
ax = Axis(fig[1,1], xlabel="Wavenumber", ylabel="Average power spectrum", xscale = log2, yscale = log10)
band!(x, real_l, real_u, color=(:orange, 0.3), label="real low resolution")
lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l, fake_u, color=(:purple, 0.3), label="generated low resolution")
lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
if ch == 1
    lines!(x, (x).^(-5/3), color=:black, linestyle=:dash)
else
    lines!(x, 10*(x).^(-3), color=:black, linestyle=:dash)
end
xlims!(ax, min_x, max_x)
ylims!(ax, min_y, max_y)

if ch == 1
    axislegend(; position= :lb, titlesize= 22)
else
    axislegend(; position= :rt, titlesize= 22)
end
save("fig:spectra_low_res_ch$(ch).png", fig, px_per_unit = 2)