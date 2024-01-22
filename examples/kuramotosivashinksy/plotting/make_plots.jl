using CairoMakie
using DataFrames
using HDF5
using Statistics
include("./distribution_plot.jl")
include("./probability_plot.jl")
include("./image_plot.jl")
kvalues = [0,2.0, 4.0]
n_avg = 16
nsamples = 640
FT = Float32
ndata = 4000
nimages = 5

basedir = "../output_all_data_sigma_max"
trainpath = joinpath(basedir, "training_$n_avg/samples.hdf5")
fid = HDF5.h5open(trainpath, "r")
df = DataFrame(train = ones(ndata),
               likelihood_ratio = FT.(fid["likelihood_ratio"][1:ndata]),
               observable = fid["observable"][1:ndata],
               k = zeros(ndata))
train_samples = fid["samples"][:,:, 1, 1:nimages];
close(fid)

for k in kvalues
    samplepath =  joinpath(basedir, "bias_$(FT(k))_n_avg_$(n_avg)_shift_true/samples.hdf5")
    fid = HDF5.h5open(samplepath, "r")
    samples_df = DataFrame(train = zeros(nsamples),
               likelihood_ratio = fid["likelihood_ratio"][1:nsamples],
               observable = fid["observable"][1:nsamples],
               k = zeros(nsamples) .+ k)
    append!(df,samples_df)
    close(fid)
end
train_obs = df[df.train  .== 1.0, "observable"];
sigmavalues = kvalues .* std(train_obs)
distribution_plot(df, joinpath(basedir, "distribution_$n_avg.png"), nsamples, kvalues, sigmavalues)
probability_plot(df, joinpath(basedir, "probability_$n_avg.png"),nsamples, kvalues[1], sigmavalues[1])


k = kvalues[end]
unbiased_samplepath =  joinpath(basedir, "bias_$(FT(0))_n_avg_$(n_avg)_shift_true/samples.hdf5")
fid = HDF5.h5open(unbiased_samplepath, "r")
unbiased_samples = fid["generated_samples"][:,:, 1, 1:nimages]
close(fid)
biased_samplepath =  joinpath(basedir, "bias_$(FT(k))_n_avg_$(n_avg)_shift_true/samples.hdf5")
fid = HDF5.h5open(biased_samplepath, "r")
biased_samples = fid["generated_samples"][:,:, 1, nimages+1:nimages*2]
close(fid)
image_plot(unbiased_samples, biased_samples, train_samples, Int(k), joinpath(basedir, "heatmap_$n_avg.png"))
