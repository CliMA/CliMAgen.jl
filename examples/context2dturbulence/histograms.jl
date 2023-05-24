using Bootstrap
using DelimitedFiles: readdlm
using KernelDensity
using Statistics
using CairoMakie
n_boot = 10000
n_grid = 200
cil = 0.99


function get_spectra_bci(data, n_boot, cil)
    bs = bootstrap(mean, data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end

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
basedir = "stats_5_22_23"
subdirs = ["baseline","baseline_smooth", "single_wn_1e-2", "single_wn_1e-3"]
names = subdirs
paths = joinpath.(basedir, subdirs)
channels = [1, 2]

for ch in channels
    mean_fig =  Figure(resolution=(1600, 400), fontsize=18)
    # mean
    if ch == 1
        min_x, max_x = -10, -1
    else
        min_x, max_x = -3e-5, 3e-5
    end


    for i in 1:4
        path = paths[i]
        @show path
        n = names[i]
        if ch == 1
            ax = Axis(mean_fig[1,i], xlabel="Spatial mean supersaturation", ylabel="Probability density", title=n)
        else
            ax = Axis(mean_fig[1,i], xlabel="Spatial mean vorticity", ylabel="Probability density", title=n)
        end
        filepaths = joinpath.(path, ["gen_pixels_ch$ch"*"_16.0.csv", "gen_statistics_ch$ch"*"_16.0.csv"])
        stat_data = readdlm(filepaths[2], ',')
        #pixel_data = readdlm(filepaths[1], ',')

        l, u = get_pdf_bci(stat_data[:,1], min_x, max_x, n_grid, n_boot, cil)
        band!(LinRange(min_x, max_x, n_grid), l, u, color=(:purple, 0.3), label=n)
        lines!(LinRange(min_x, max_x, n_grid), l, color=(:purple, 0.5), strokewidth = 1.5)
        lines!(LinRange(min_x, max_x, n_grid), u, color=(:purple, 0.5), strokewidth = 1.5)
        filepaths = joinpath.(joinpath.(basedir, "truth"), ["train_pixels_ch$ch"*"_16.0.csv", "train_statistics_ch$ch"*"_16.0.csv"])
        stat_data = readdlm(filepaths[2], ',')
        #pixel_data = readdlm(filepaths[1], ',')

        real_l, real_u = get_pdf_bci(stat_data[:,1], min_x, max_x, n_grid, n_boot, cil)
        band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="truth")
        lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
        lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
    end
    CairoMakie.xlims!(ax, min_x, max_x)
    save("fig:mean_ch$(ch).png", mean_fig, px_per_unit = 2)
end



# Sigmas


for ch in channels
    std_fig =  Figure(resolution=(1600, 400), fontsize=18)
    min_x, max_x = 0, 10
    for i in 1:4
        path = paths[i]
        @show path
        n = names[i]
        if ch == 1
            ax = Axis(std_fig[1,i], xlabel="Spatial mean supersaturation", ylabel="Probability density", title=n)
        else
            ax = Axis(std_fig[1,i], xlabel="Spatial mean vorticity", ylabel="Probability density", title=n)
        end
        filepaths = joinpath.(path, ["gen_pixels_ch$ch"*"_16.0.csv", "gen_statistics_ch$ch"*"_16.0.csv"])
        stat_data = readdlm(filepaths[2], ',')
        #pixel_data = readdlm(filepaths[1], ',')

        l, u = get_pdf_bci(sqrt.(stat_data[:,2]), min_x, max_x, n_grid, n_boot, cil)
        band!(LinRange(min_x, max_x, n_grid), l, u, color=(:purple, 0.3), label=n)
        lines!(LinRange(min_x, max_x, n_grid), l, color=(:purple, 0.5), strokewidth = 1.5)
        lines!(LinRange(min_x, max_x, n_grid), u, color=(:purple, 0.5), strokewidth = 1.5)
        filepaths = joinpath.(joinpath.(basedir, "truth"), ["train_pixels_ch$ch"*"_16.0.csv", "train_statistics_ch$ch"*"_16.0.csv"])
        stat_data = readdlm(filepaths[2], ',')
        #pixel_data = readdlm(filepaths[1], ',')

        real_l, real_u = get_pdf_bci(sqrt.(stat_data[:,2]), min_x, max_x, n_grid, n_boot, cil)
        band!(LinRange(min_x, max_x, n_grid), real_l, real_u, color=(:orange, 0.3), label="truth")
        lines!(LinRange(min_x, max_x, n_grid), real_l, color=(:orange, 0.5), strokewidth = 1.5)
        lines!(LinRange(min_x, max_x, n_grid), real_u, color=(:orange, 0.5), strokewidth = 1.5)
    end
    CairoMakie.xlims!(ax, min_x, max_x)
    save("fig:std_ch$(ch).png", std_fig, px_per_unit = 2)
end

# Spectra
for ch in channels
    spectra_fig = Figure(resolution=(1600, 400), fontsize=24)
    n_boot = 10000
    cil = 0.99
    x = (1:256)
    min_x = 2^0
    max_x = 2^8
    min_y = 1e-10
    max_y = ch == 1 ? 1e0 : 1e1 
    for i in 1:4
        path = paths[i]
        @show path
        n = names[i]
        ax = Axis(spectra_fig[1,i], xlabel="Wavenumber", ylabel="Average power spectrum", title=n, xscale = log2, yscale = log10)
        filepaths = joinpath.(path, ["gen_pixels_ch$ch"*"_16.0.csv", "gen_statistics_ch$ch"*"_16.0.csv"])
        stat_data = readdlm(filepaths[2], ',')
        #pixel_data = readdlm(filepaths[1], ',')
        if ch ==1
            spectra = [Array(stat_data[k,5:end-1]) for k in 1:size(stat_data)[1]]
        else
            spectra =[Array(stat_data[k,5:end]) for k in 1:size(stat_data)[1]]
        end
        l, u = get_spectra_bci(spectra, n_boot, cil)
        band!(x, l, u, color=(:purple, 0.3), label=n)
        lines!(x, l, color=(:purple, 0.5), strokewidth = 1.5)
        lines!(x, u, color=(:purple, 0.5), strokewidth = 1.5)
        filepaths = joinpath.(joinpath.(basedir, "truth"), ["train_pixels_ch$ch"*"_16.0.csv", "train_statistics_ch$ch"*"_16.0.csv"])
        stat_data = readdlm(filepaths[2], ',')
        #pixel_data = readdlm(filepaths[1], ',')
        if ch ==1
            spectra = [Array(stat_data[k,5:end-1]) for k in 1:size(stat_data)[1]]
        else
            spectra =[Array(stat_data[k,5:end]) for k in 1:size(stat_data)[1]]
        end
        l, u = get_spectra_bci(spectra, n_boot, cil)
        band!(x, l, u, color=(:orange, 0.3), label="truth")
        lines!(x, l, color=(:orange, 0.5), strokewidth = 1.5)
        lines!(x, u, color=(:orange, 0.5), strokewidth = 1.5)
        CairoMakie.xlims!(ax, min_x, max_x)
        CairoMakie.ylims!(ax, min_y, max_y)
    end
    save("fig:spectra_ch$(ch).png", spectra_fig, px_per_unit = 2)
end


# Condensation rate
fig = Figure(resolution=(1600, 400), fontsize=24)
ch = 1
n_pixels = 1600000
n_boot = 1000
n_grid = 100
cil = 0.99
FT = Float32
min_x, max_x = 0, 325
min_y, max_y = 1e-7, 1e-1
x = LinRange(min_x, max_x, n_grid)
τ = 1e-2
filepaths = joinpath.(joinpath.(basedir, "truth"), ["train_pixels_ch$ch"*"_16.0.csv", "train_statistics_ch$ch"*"_16.0.csv"])
pixel_data = readdlm(filepaths[1], ',')
cr = pixel_data[1:n_pixels]
cr = cr[cr .> 0] / τ
l_real, u_real = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)
for i in 1:4
    path = paths[i]
    @show path
    n = names[i]
    ax = Axis(fig[1,i], xlabel="Condensation rate", ylabel="Probability density", title=n, yscale=log10, titlefont = :regular)
    filepaths = joinpath.(path, ["gen_pixels_ch$ch"*"_16.0.csv", "gen_statistics_ch$ch"*"_16.0.csv"])
    pixel_data = readdlm(filepaths[1], ',')
    cr = pixel_data[1:n_pixels]
    cr = cr[cr .> 0] / τ
    l, u = get_pdf_bci(cr, min_x, max_x, n_grid, n_boot, cil)

    band!(x, l, u, color=(:purple, 0.3), label=n)
    lines!(x, l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(x, u, color=(:purple, 0.5), strokewidth = 1.5)

    band!(x, l_real, u_real, color=(:orange, 0.3), label="truth")
    lines!(x, l_real, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, u_real, color=(:orange, 0.5), strokewidth = 1.5)
    CairoMakie.xlims!(ax, min_x, max_x)
    CairoMakie.ylims!(ax, min_y, max_y)
end
save("fig:cond_rate.png", fig, px_per_unit = 2)
