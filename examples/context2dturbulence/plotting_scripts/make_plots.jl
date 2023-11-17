using Bootstrap
using CairoMakie
using DataFrames
using DelimitedFiles: readdlm
using Interpolations
using KernelDensity
using Statistics

## loading statistics into dataframes
types = [:train, :gen]
channels = [1, 2]
wavenumbers = [0.0, 2.0, 4.0, 8.0, 16.0]

# extract data from files
data = []
for type in types
    for ch in channels
        for wn in wavenumbers
            if wn == 0.0 && type == :gen
                nothing
            else

                if wn == 0.0 || type == :train
                    filename = "/groups/esm/kdeck/downscaling/stats/data/$(type)/$(type)_statistics_ch$(ch)_$(wn).csv"
                else
                    filename = "/groups/esm/kdeck/downscaling/stats/data/$(type)/downscale_$(type)_statistics_ch$(ch)_$(wn).csv"
                end
                df = DataFrame(readdlm(filename, ',', Float32, '\n'), :auto)
                # no condensation rate for vorticity
                if ch == 2
                    df.x261 .= nothing
                end
                df.isreal .= type == :gen ? false : true
                df.channel .= ch
                df.wavenumber .= wn
                # adjust number of observations, they are different for the different sets
                if type == :train
                    df = df[end-99:end, :]
                end
                push!(data, df)
            end
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

# extract pixels from files
pixels = []
for type in types
    for ch in channels
        for wn in wavenumbers
            if wn == 0.0 && type == :gen
                nothing
            else
                if type == :train
                    filename = "/groups/esm/kdeck/downscaling/stats/data/$(type)/$(type)_pixels_ch$(ch)_$(wn).csv"
                else
                    filename = "/groups/esm/kdeck/downscaling/stats/data/$(type)/downscale_$(type)_pixels_ch$(ch)_$(wn).csv"
                end

                println(filename)
                df = DataFrame(readdlm(filename, ',', Float32, '\n')[:][1:1600000,:]', :auto)
                df.isreal .= type == :gen ? false : true
                df.channel .= ch
                df.wavenumber .= wn
                push!(pixels, df)
            end
        end
    end
end
pixels = vcat(pixels...)

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

function get_spectra_bci(data, n_boot, cil)
    bs = bootstrap(mean, data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end

function get_mean_bci(data, n_boot, cil)
    bs = bootstrap(mean, data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end

function percentile_scale(x, data, lower=1.0, upper=4.0)
    expo = LinRange(lower, upper, length(x))
    quantiles = map(x -> 1-1/10^x, expo)
    x_quantiles = map(x -> quantile(data, x), quantiles)
    interp_linear = linear_interpolation(x_quantiles, quantiles)
    return interp_linear.(x)
end

## plotting
include("plot_pixelwise_corr.jl")
ch = 1
include("plot_mean.jl")
include("plot_pdfs.jl")
include("plot_spectra.jl")
include("plot_cond_rate.jl")

ch = 2
include("plot_mean.jl")
include("plot_pdfs.jl")
include("plot_spectra.jl")
