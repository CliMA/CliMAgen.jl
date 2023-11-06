# fig
fig = Figure(resolution=(1600, 400), fontsize=24)
n_pixels = 1600000
n_grid = 100
cil = 0.99
FT = Float64
#min_x, max_x = -25, 5
min_x, max_x = 0, 3.5
x = LinRange(min_x, max_x, 100)
amp=0.1

ch = 1
wn = 2.0
cr_real = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_real = cr_real .+ randn(n_pixels)*amp
cr_real = cr_real[cr_real .> 0]
cr_fake = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
# cr_fake = vcat([cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,4)], cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,4)], cr_fake[1:div(n_pixels,8)]]...)
cr_fake = cr_fake[cr_fake .> 0]
cr_lr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_lr = cr_lr[cr_lr .> 0]
pdf_real = get_pdf(cr_real, min_x, max_x, n_grid) .+ eps(FT)
pdf_fake = get_pdf(cr_fake, min_x, max_x, n_grid) .+ eps(FT)
pdf_lr = get_pdf(cr_lr, min_x, max_x, n_grid) .+ eps(FT)

ax = Axis(fig[1,1], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn", yscale=log10)
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)
ylims!(ax, 1e-5, 1e1)

wn = 4.0
cr_real = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_real = cr_real .+ randn(n_pixels)*amp
cr_real = cr_real[cr_real .> 0]
cr_fake = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_fake = cr_fake[cr_fake .> 0]
cr_lr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_lr = cr_lr[cr_lr .> 0]
pdf_real = get_pdf(cr_real, min_x, max_x, n_grid) .+ eps(FT)
pdf_fake = get_pdf(cr_fake, min_x, max_x, n_grid) .+ eps(FT)
pdf_lr = get_pdf(cr_lr, min_x, max_x, n_grid) .+ eps(FT)

ax = Axis(fig[1,2], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn", yscale=log10)
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)
ylims!(ax, 1e-5, 1e1)

wn = 8.0
cr_real = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_real = cr_real .+ randn(n_pixels)*amp
cr_real = cr_real[cr_real .> 0]
cr_fake = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_fake = cr_fake[cr_fake .> 0]
cr_lr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_lr = cr_lr[cr_lr .> 0]
pdf_real = get_pdf(cr_real, min_x, max_x, n_grid) .+ eps(FT)
pdf_fake = get_pdf(cr_fake, min_x, max_x, n_grid) .+ eps(FT)
pdf_lr = get_pdf(cr_lr, min_x, max_x, n_grid) .+ eps(FT)

ax = Axis(fig[1,3], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn", yscale=log10)
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)
ylims!(ax, 1e-5, 1e1)

wn = 16.0
cr_real = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_real = cr_real .+ randn(n_pixels)*amp
cr_real = cr_real[cr_real .> 0]
cr_fake = Array(pixels[(pixels.isreal .== false) .&& (pixels.wavenumber .== wn) .&& (pixels.channel .== ch), 1:n_pixels])[:]
# cr_fake = vcat([cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,4)], cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,8)], cr_fake[1:div(n_pixels,4)], cr_fake[1:div(n_pixels,8)]]...)
cr_fake = cr_fake[cr_fake .> 0]
cr_lr = Array(pixels[(pixels.isreal .== true) .&& (pixels.wavenumber .== 0.0) .&& (pixels.channel .== ch), 1:n_pixels])[:]
cr_lr = cr_lr[cr_lr .> 0]
pdf_real = get_pdf(cr_real, min_x, max_x, n_grid) .+ eps(FT)
pdf_fake = get_pdf(cr_fake, min_x, max_x, n_grid) .+ eps(FT)
pdf_lr = get_pdf(cr_lr, min_x, max_x, n_grid) .+ eps(FT)

ax = Axis(fig[1,4], xlabel="Supersaturation", ylabel="Probability density", title="k = $wn", yscale=log10)
plot!(x, pdf_real, color=:black)
plot!(x, pdf_fake, color=:blue)
plot!(x, pdf_lr, color=:green)
ylims!(ax, 1e-5, 1e1)

save("fig:pdfs_check_2.png", fig, px_per_unit = 2)