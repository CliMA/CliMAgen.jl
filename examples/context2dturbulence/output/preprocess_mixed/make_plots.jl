using CairoMakie
using DataFrames
using DelimitedFiles: readdlm

# loading data into dataframes
# extract data from files
types = [:train, :gen]
channels = [1]
wavenumbers = [1.0, 2.0, 4.0, 8.0, 16.0]
data = []
for type in types
    for ch in channels
        for wn in wavenumbers
            filename = "gen_statistics_ch$(ch)_$(wn).csv"
            df = DataFrame(readdlm(filename, ',', Float64, '\n'), :auto)
            df.isreal .= type == :gen ? false : true
            df.channel .= ch
            df.wavenumber .= wn
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
