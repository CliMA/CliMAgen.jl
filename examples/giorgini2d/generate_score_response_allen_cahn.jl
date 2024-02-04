using Statistics
using Random
using ProgressBars
using TOML
using Plots
using StatsBase
using HDF5
using BSON
using CliMAgen
using CliMAgen: expand_dims, MeanSpatialScaling, StandardScaling, apply_preprocessing, invert_preprocessing
using Distributions
using CUDA
using Flux 
using GLMakie
using FFTW

# run from giorgini2d
package_dir = pwd()
include("./trajectory_utils.jl")

FT = Float32
experiment_toml = "Experiment.toml"
model_toml = "Model.toml"

toml_dict = TOML.parsefile(model_toml)
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
α = FT(toml_dict["param_group"]["alpha"])
β = FT(toml_dict["param_group"]["beta"])
γ = FT(toml_dict["param_group"]["gamma"])
σ = FT(toml_dict["param_group"]["sigma"])
savedir = "$(params.experiment.savedir)_$(α)_$(β)_$(γ)_$(σ)"
f_path = "data/data_$(α)_$(β)_$(γ)_$(σ).hdf5"

@info "loading data"
fid = h5open(f_path, "r")
trj = read(fid, "timeseries")
timeseries_shape = read(fid["timeseries shape"] )
M, N, _, L = size(trj)
close(fid)
# Compute using the score function
@info "loading score function"
checkpoint_path = joinpath(savedir, "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth
dev = Flux.gpu
model = dev(model_smooth)
t0 = 0.0

@info "applying score function"
batchsize = 32
nbatches = floor(Int, L / batchsize)
# need to apply in batchesc
scores = zeros(Float32, M, N, 1, L)
for i in ProgressBar(1:nbatches)
    x = dev(CuArray(trj[:, :, [1], (i-1)*batchsize+1:i*batchsize]))
    scores[:, :, 1, (i-1)*batchsize+1:i*batchsize] = Array(CliMAgen.score(model, x, Float32(0.0)))
end
@info "leftover"
if nbatches * batchsize < L
    x = dev(CuArray(trj[:, :, [1], (nbatches*batchsize+1):L]))
    scores[:, :, 1, (nbatches*batchsize+1):L] = Array(CliMAgen.score(model, x, Float32(0.0)))
end

pixel_value = reshape(trj, (timeseries_shape[1], timeseries_shape[2], timeseries_shape[4], timeseries_shape[5]))
score_values = reshape(scores, (timeseries_shape[1], timeseries_shape[2], timeseries_shape[4], timeseries_shape[5]))


function score_response_function(pixel_value, score_values; skip = 1)
    N = size(pixel_value)[1]
    Ne = size(pixel_value)[3]
    endindex = size(pixel_value[:, :, :, 1:skip:end])[end]÷2
    score_response_array = zeros(N, N, endindex); 
    @info "Computing Response Function"
    for i in ProgressBar(1:Ne)    
        # The N^2 is due to fft stuff
        # The "two" is due to the Fokker-Planck equation having that factor of 1/2 in front of the diffusion term
        score_response_array .+= -real.(ifft(fft(pixel_value[:, :, i, 1:skip:end]) .* ifft(score_values[:, :, i, 1:skip:end]))[:, :, 1:endindex]/Ne)
    end
    return score_response_array
end

function hack(score_response_array)
    N, N, endindex = size(score_response_array)
    hack_response = zeros(N^2, N^2, endindex);
    @info "Constructing Hack Matrix"
    for k in ProgressBar(1:endindex)
        for i in 1:N^2
            ii = (i-1) % N + 1
            jj = (i-1) ÷ N + 1
            @inbounds hack_response[:, i, k] .= circshift(score_response_array[:, :, k], (ii-1, jj-1))[:]
        end
    end
    @info "Inverting Hack Matrix"
    C⁻¹ = inv(hack_response[:, :, 1])
    # hack_score_response = [(hack_response[:, :, i] * C⁻¹ + C⁻¹ * hack_response[:, :, i])/2  for i in 1:endindex]
    @info "Constructing Hack Response"
    return [hack_response[:, :, i] * C⁻¹ for i in ProgressBar(1:endindex)]
end

sra = score_response_function(pixel_value, score_values)
hsra = hack(sra)

##
fig = Figure(resolution = (772, 209))
N = 32
lw = 3
ts = collect(0:length(hsra)-1)
for i in 1:4
    indexchoice = i
    ax = GLMakie.Axis(fig[1, i]; title = "1 -> " * string(indexchoice), xlabel = "time", ylabel = "response")
    lines!(ax, ts, [reshape(hsra[k][:, 1], (N, N))[indexchoice, 1] for k in eachindex(hsra)], color = (:red, 0.4), linewidth = lw, label = "score")
    # scatter!(ax, ts[1:size(nr[indexchoice, 1, 1:32:end])[end]], nr[indexchoice, 1, 1:32:end], color = (:orange, 0.4), linewidth = lw, label = "perturbation")
    if i == 1
        axislegend(ax, position = :rt)
    else
        hideydecorations!(ax)
    end
    GLMakie.xlims!(ax, (0, 50))
    GLMakie.ylims!(ax, (-0.15, 1.1))
end
display(fig)
##
Nt = size(hsra)[end]
climagen_score = zeros(N, Nt)
for i in 1:N
    climagen_score[i, :] .= [reshape(hsra[k][:, 1], (N, N))[i, 1] for k in eachindex(hsra)]
end

generative_response = zeros(N, N, Nt)
for i in 1:N, j in 1:N
    generative_response[i, j, :] .= [reshape(hsra[k][:, 1], (N, N))[i, j] for k in eachindex(hsra)]
end

hfile = h5open(f_path[1:end-5] * "_generative_response.hdf5", "w")
hfile["generative response"] = generative_response
hfile["generative response no hack"] = sra
close(hfile)
