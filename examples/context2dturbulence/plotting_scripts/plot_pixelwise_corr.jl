# fig
fig = Figure(resolution=(1600, 400), fontsize=24)
n_pixels = 1600000
n_boot = 10000
n_grid = 100
cil = 0.99
FT = Float64
min_x = -5
max_x = 5
x = LinRange(min_x, max_x, 100)

wn = 2.0
ch1 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
real_l_hr, real_u_hr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
ch1 = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
fake_l_hr, fake_u_hr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
ch1 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
real_l_lr, real_u_lr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT)
ax = Axis(fig[1,1], xlabel="Pixel correlation", ylabel="Probability density", title=L"k_x = k_y = 2", yscale=log10)

band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high res.")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-4, 1e0)
#axislegend(; position= :lb, titlesize= 22)

wn = 4.0
ch1 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
real_l_hr, real_u_hr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
ch1 = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
fake_l_hr, fake_u_hr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
ch1 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
real_l_lr, real_u_lr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT)
ax = Axis(fig[1,1], xlabel="Pixel correlation", ylabel="Probability density", title=L"k_x = k_y = 4", yscale=log10)

band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high res.")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-4, 1e0)
#axislegend(; position= :lb, titlesize= 22)

wn = 8.0
ch1 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
real_l_hr, real_u_hr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
ch1 = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
fake_l_hr, fake_u_hr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
ch1 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
real_l_lr, real_u_lr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT)
ax = Axis(fig[1,1], xlabel="Pixel correlation", ylabel="Probability density", title=L"k_x = k_y = 8", yscale=log10)

band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high res.")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-4, 1e0)

wn = 16.0
ch1 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
real_l_hr, real_u_hr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
ch1 = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
fake_l_hr, fake_u_hr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
ch1 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== 1), 1:n_pixels])[:]
ch2 = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== 2), 1:n_pixels])[:]
cor12 = (ch1 .- mean(ch1)) .* (ch2 .- mean(ch2)) ./(std(ch1)*std(ch2))#Statistics.cor(ch1, ch2)
real_l_lr, real_u_lr = get_pdf_bci(cor12, min_x, max_x, n_grid, n_boot, cil)
real_l_hr, real_u_hr, fake_l_hr, fake_u_hr = real_l_hr .+ eps(FT), real_u_hr .+ eps(FT), fake_l_hr .+ eps(FT), fake_u_hr .+ eps(FT)
real_l_lr, real_u_lr = real_l_lr .+ eps(FT), real_u_lr.+ eps(FT)
ax = Axis(fig[1,1], xlabel="Pixel correlation", ylabel="Probability density", title=L"k_x = k_y = 16", yscale=log10)

band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
band!(x, fake_l_hr, fake_u_hr, color=(:purple, 0.3), label="generated high res.")
lines!(x, fake_l_hr, color=(:purple, 0.5), strokewidth = 1.5)
lines!(x, fake_u_hr, color=(:purple, 0.5), strokewidth = 1.5)
band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
xlims!(ax, min_x, max_x)
ylims!(ax, 1e-4, 1e0)
if ch == 1
    axislegend(; position= :lt, labelsize= 16)
else
    axislegend(; position= :cb, labelsize= 16)
end

save("fig:pixelwise_corr.png", fig, px_per_unit = 2)
