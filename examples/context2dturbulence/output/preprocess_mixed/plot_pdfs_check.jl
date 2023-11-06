# fig
fig = Figure(resolution=(1600, 400), fontsize=24)
n_pixels = 1600000
n_grid = 100
cil = 0.99
FT = Float64
min_x, max_x = -25, 5
x = LinRange(min_x, max_x, 100)
amp=0

ch = 1
wn = 2.0
cr_real = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_real = cr_real .+ randn(n_pixels)*amp
cr_fake = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_lr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
pdf_real = get_pdf(cr_real, min_x, max_x, n_grid)
pdf_fake = get_pdf(cr_fake, min_x, max_x, n_grid)
pdf_lr = get_pdf(cr_lr, min_x, max_x, n_grid)

ax = Axis(fig[1,1], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn")
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)

wn = 4.0
cr_real = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_real = cr_real .+ randn(n_pixels)*amp
cr_fake = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_lr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
pdf_real = get_pdf(cr_real, min_x, max_x, n_grid)
pdf_fake = get_pdf(cr_fake, min_x, max_x, n_grid)
pdf_lr = get_pdf(cr_lr, min_x, max_x, n_grid)

ax = Axis(fig[1,2], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn")
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)

wn = 8.0
cr_real = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_real = cr_real .+ randn(n_pixels)*amp
cr_fake = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_lr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
pdf_real = get_pdf(cr_real, min_x, max_x, n_grid)
pdf_fake = get_pdf(cr_fake, min_x, max_x, n_grid)
pdf_lr = get_pdf(cr_lr, min_x, max_x, n_grid)

ax = Axis(fig[1,3], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn")
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)

wn = 16.0
cr_real = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_real = cr_real .+ randn(n_pixels)*amp
cr_fake = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_lr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
pdf_real = get_pdf(cr_real, min_x, max_x, n_grid)
pdf_fake = get_pdf(cr_fake, min_x, max_x, n_grid)
pdf_lr = get_pdf(cr_lr, min_x, max_x, n_grid)

ax = Axis(fig[1,4], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn")
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)
save("fig:pdfs_check.png", fig, px_per_unit = 2)