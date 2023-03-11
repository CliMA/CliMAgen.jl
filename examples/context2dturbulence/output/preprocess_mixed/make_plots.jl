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
                df = df[end-99:end, :]
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

struct IntegerTicks end

## plotting
include("plot_pdfs.jl")
#include("plot_spectra.jl")
