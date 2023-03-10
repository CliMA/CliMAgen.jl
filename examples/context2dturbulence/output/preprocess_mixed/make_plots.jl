using Bootstrap
using CairoMakie
using DataFrames
using DelimitedFiles: readdlm
using KernelDensity
using Statistics

## loading data into dataframes
# extract data from files
types = [:train, :gen]
channels = [1]
wavenumbers = [1.0, 2.0, 4.0, 8.0, 16.0]
data = []
for type in types
    for ch in channels
        for wn in wavenumbers
            filename = "$(type)_statistics_ch$(ch)_$(wn).csv"
            df = DataFrame(readdlm(filename, ',', Float64, '\n'), :auto)
            df.isreal .= type == :gen ? false : true
            df.channel .= ch
            df.wavenumber .= wn
            if type == :train
                df = df[1:100, :]
            end
            push!(data, df)
        end
    end
end
data = vcat(data...)

# headers
hd_stats = [:mean, :variance, :skewness, :kurtosis]
hd_spec = []
hd_cond = [:cond_rate]
rename!(data, Dict(Symbol("x$i") => s for (i, s) in enumerate(hd_stats)))
rename!(data, Dict(Symbol("x$i") => Symbol("s$(i-4)") for i in 5:260))
rename!(data, Dict(:x261 => hd_cond...))

# split it up!
stats = data[:, vcat(hd_stats, [:isreal, :channel, :wavenumber])]
spectra = data[:, vcat([Symbol("s$i") for i in 1:256], [:isreal, :channel, :wavenumber])]
cond = data[:, vcat(hd_cond, [:isreal, :channel, :wavenumber])]


## plotting
n_boot = 10000
n_grid = 200
cil = 0.99

# PDFs
pdfs = Figure(resolution=(1600, 2000))

# mean
min_x, max_x = -10, -1
wn = 1.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[1,1], ylabel="Probability distribution")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)
axislegend(; position= :lt, titlesize= 22)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[1,2])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[1,3], xlabel="Spatial mean")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[1,4])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].mean, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[1,5])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="real high resolution")
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5, label="generated high resolution")
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

# variance
min_x, max_x = 0, 50
wn = 1.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[2,1], ylabel="Probability distribution")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[2,2])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[2,3], ylabel="Spatial variance")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[2,4])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].variance, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[2,5])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

# skewnewss
min_x, max_x = -300, 50
wn = 1.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[3,1], ylabel="Probability distribution")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[3,2])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[3,3], xlabel="Spatial skewness")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[3,4])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].skewness, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[3,5])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

# kurtosis
min_x, max_x = -1000, 3000
wn = 1.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[4,1], ylabel="Probability distribution")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 2.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[4,2])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 4.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[4,3], xlabel="Spatial kurtosis")
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 8.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[4,4])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

wn = 16.0
real_l, real_u = get_pdf_bci(stats[(stats.isreal .== true) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
fake_l, fake_u = get_pdf_bci(stats[(stats.isreal .== false) .&& (stats.wavenumber .== wn), :].kurtosis, min_x, max_x, n_grid, n_boot, cil)
ax = Axis(pdfs[4,5])
band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
band!(LinRange(min_x, max_x, n_grid), fake_l, fake_u, color=(:purple, 0.3), strokecolor = :grey20, strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_l, color=(:purple, 0.5), strokewidth = 1.5)
lines!(LinRange(min_x, max_x, n_grid), fake_u, color=(:purple, 0.5), strokewidth = 1.5)

save("fig1:pdfs.png", pdfs, px_per_unit = 2) # size = 1600 x 1200 px


## utils 
function get_pdf(data, min_x, max_x, n_grid)
    estimate = kde(data)
    pdf(estimate, LinRange(min_x, max_x, n_grid))
end

function get_pdf_bci(data, min_x, max_x, n_grid, n_boot, cil)
    x = LinRange(min_x, max_x, n_grid)
    bs = bootstrap(x -> get_pdf(x, min_x, max_x, n_grid), data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end