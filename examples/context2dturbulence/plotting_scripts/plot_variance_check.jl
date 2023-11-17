# fig
fig = Figure(resolution=(1600, 400), fontsize=24)
n_pixels = 1600000
n_grid = 100
cil = 0.99
FT = Float64
min_x, max_x = 0, 10
x = LinRange(min_x, max_x, 100)
amp=0

ch = 1
wn = 2.0
cr_real = Array(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].variance)[:]
cr_fake = Array(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].variance)[:]
cr_lr = Array(stats[(stats.isreal .== true) .&& (stats.wavenumber .== 0.0) .&& (stats.channel .== ch), :].variance)[:]
pdf_real = get_pdf(sqrt.(cr_real), min_x, max_x, n_grid)
pdf_fake = get_pdf(sqrt.(cr_fake), min_x, max_x, n_grid)
pdf_lr = get_pdf(sqrt.(cr_lr), min_x, max_x, n_grid)

ax = Axis(fig[1,1], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn")
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)

wn = 4.0
cr_real = Array(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].variance)[:]
cr_fake = Array(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].variance)[:]
cr_lr = Array(stats[(stats.isreal .== true) .&& (stats.wavenumber .== 0.0) .&& (stats.channel .== ch), :].variance)[:]
pdf_real = get_pdf(sqrt.(cr_real), min_x, max_x, n_grid)
pdf_fake = get_pdf(sqrt.(cr_fake), min_x, max_x, n_grid)
pdf_lr = get_pdf(sqrt.(cr_lr), min_x, max_x, n_grid)

ax = Axis(fig[1,2], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn")
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)

wn = 8.0
cr_real = Array(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].variance)[:]
cr_fake = Array(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].variance)[:]
cr_lr = Array(stats[(stats.isreal .== true) .&& (stats.wavenumber .== 0.0) .&& (stats.channel .== ch), :].variance)[:]
pdf_real = get_pdf(sqrt.(cr_real), min_x, max_x, n_grid)
pdf_fake = get_pdf(sqrt.(cr_fake), min_x, max_x, n_grid)
pdf_lr = get_pdf(sqrt.(cr_lr), min_x, max_x, n_grid)

ax = Axis(fig[1,3], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn")
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)

wn = 16.0
cr_real = Array(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].variance)[:]
cr_fake = Array(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn) .&& (stats.channel .== ch), :].variance)[:]
cr_lr = Array(stats[(stats.isreal .== true) .&& (stats.wavenumber .== 0.0) .&& (stats.channel .== ch), :].variance)[:]
pdf_real = get_pdf(sqrt.(cr_real), min_x, max_x, n_grid)
pdf_fake = get_pdf(sqrt.(cr_fake), min_x, max_x, n_grid)
pdf_lr = get_pdf(sqrt.(cr_lr), min_x, max_x, n_grid)

ax = Axis(fig[1,4], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn")
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)

save("fig:variance_check.png", fig, px_per_unit = 2)