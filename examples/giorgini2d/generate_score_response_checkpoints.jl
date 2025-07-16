using Statistics
using Random
using ProgressBars
using TOML
using HDF5
using BSON
using CliMAgen
using CliMAgen: expand_dims, MeanSpatialScaling, StandardScaling, apply_preprocessing, invert_preprocessing
using Distributions
using CUDA
using Flux 
using UnicodePlots
# using GLMakie
using FFTW

# run from giorgini2d
package_dir = pwd()
include("./trajectory_utils.jl")

# /home/sandre/orcd/scratch/ResponseFunctionRestartFiles/output_9_full_ns_large_kernels/
# data_9.0_0.0_0.0_0.0.hdf5
FT = Float32

#=
toml_dict = TOML.parsefile(model_toml)
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
α = FT(toml_dict["param_group"]["alpha"])
β = FT(toml_dict["param_group"]["beta"])
γ = FT(toml_dict["param_group"]["gamma"])
σ = FT(toml_dict["param_group"]["sigma"])
=#
# savedir = "/nobackup1/sandre/ResponseFunctionRestartFiles/$(params.experiment.savedir)_$(α)_$(β)_$(γ)_$(σ)"
# f_path  = "/nobackup1/sandre/ResponseFunctionTrainingData/data_$(α)_$(β)_$(γ)_$(σ).hdf5"
# f_path = "/nobackup1/sandre/ResponseFunctionTrainingData/data_4.0_$(β)_$(γ)_$(σ).hdf5"
savedir = "/home/sandre/orcd/scratch/ResponseFunctionRestartFiles/output_9_full_ns_large_kernels"
# savedir = "/home/sandre/orcd/scratch/ResponseFunctionRestartFiles/output_9.0_0.0_0.0_0.0"
f_path = "/home/sandre/orcd/scratch/ResponseFunctionTrainingData/data_9.0_0.0_0.0_0.0.hdf5"
f_path = "/orcd/data/raffaele/001/sandre/Repositories/ResponseFunction/FourierCore/new_data.hdf5"

@info "loading data"
fid = h5open(f_path, "r")
trj = read(fid, "timeseries")
trj_delta = read(fid["timeseries_δ"])
timeseries_shape = read(fid["timeseries shape"] )
traj = reshape(trj, (timeseries_shape[1], timeseries_shape[2], timeseries_shape[4], timeseries_shape[5]))
traj_delta = reshape(trj_delta, (timeseries_shape[1], timeseries_shape[2], timeseries_shape[4], timeseries_shape[5]))
delta = read(fid, "delta")
M, N, _, L = size(trj)
close(fid)
# Compute using the score function
sras = []
sras_2 = []
sras_3 = []
sras_4 = []
sras_5 = []
for checkpoint_index in ProgressBar(1300:100:2000)
@info "loading score function"
    checkpoint_path = joinpath(savedir, "checkpoint_$checkpoint_index.bson")
    if isfile(checkpoint_path)
        checkpoint_path = checkpoint_path
    else
        checkpoint_path = joinpath(savedir, "checkpoint_$(Float64(checkpoint_index)).bson")
    end
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


    function score_response_function(pixel_value, score_values; skip = 1, pow = 1)
        N = size(pixel_value)[1]
        Ne = size(pixel_value)[3]
        endindex = size(pixel_value[:, :, :, 1:skip:end])[end]÷2
        score_response_array = zeros(N, N, endindex); 
        @info "Computing Response Function"
        for i in ProgressBar(1:Ne)    
            # The N^2 is due to fft stuff
            # The "two" is due to the Fokker-Planck equation having that factor of 1/2 in front of the diffusion term
            score_response_array .+= -real.(ifft(fft(pixel_value[:, :, i, 1:skip:end] .^pow ) .* ifft(score_values[:, :, i, 1:skip:end]))[:, :, 1:endindex]/Ne)
        end
        return score_response_array
    end

    function hack_matrix(score_response_array)
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
        return C⁻¹
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
    sra_2 = score_response_function(pixel_value, score_values, pow = 2)
    sra_3 = score_response_function(pixel_value, score_values, pow = 3)
    sra_4 = score_response_function(pixel_value, score_values, pow = 4)
    sra_5 = score_response_function(pixel_value, score_values, pow = 5)
    C⁻¹ = hack_matrix(sra)

    
    #=
    hsra = hack(sra)
    generative_response = zeros(N, N, length(hsra))
    for i in 1:N, j in 1:N
        generative_response[i, j, :] .= [reshape(hsra[k][:, 1], (N, N))[i, j] for k in eachindex(hsra)]
    end
    =#
    
    hfile = h5open("/orcd/data/raffaele/001/sandre/Repositories/ResponseFunction/CliMAgen.jl/"* "new_generative_response_$checkpoint_index.hdf5", "w")
    # hfile["generative response"] = generative_response
    hfile["score values"] = score_values
    hfile["pixel value"] = pixel_value
    hfile["generative response no hack"] = sra
    hfile["generative response no hack 2"] = sra_2
    hfile["generative response no hack 3"] = sra_3
    hfile["generative response no hack 4"] = sra_4
    hfile["generative response no hack 5"] = sra_5
    hfile["hack matrix"] = C⁻¹
    close(hfile)
    push!(sras, sra)
    push!(sras_2, sra_2)
    push!(sras_3, sra_3)
    push!(sras_4, sra_4)
    push!(sras_5, sra_5)
end


for i in eachindex(sras)
    println("First moment")
    println(sras[i][1,1,1])
    println(1)

    println("Second moment")
    println(sras_2[i][1,1,1])
    println(2 * mean(trj ))

    println("Third moment")
    println(sras_3[i][1,1,1])
    println(mean(trj .^2) * 3)

    println("Fourth moment")
    println(sras_4[i][1,1,1])
    println(mean(trj .^3) * 4)

    println("Fifth moment")
    println(sras_5[i][1,1,1])
    println(mean(trj .^4)*5)
    println((var(trj)^2 * 3 ) * 5)
    println("--------------------------------")
end

scale_factor = 5
first_moment = mean(traj_delta - traj, dims = 3)/delta * scale_factor ;
second_moment = mean(traj_delta .^2 - traj .^2, dims = 3)/delta * scale_factor;
third_moment = mean(traj_delta .^3 - traj .^3, dims = 3)/delta * scale_factor;
fourth_moment = mean(traj_delta .^4 - traj .^4, dims = 3)/delta * scale_factor;
fifth_moment = mean(traj_delta .^5 - traj .^5, dims = 3)/delta * scale_factor;

scp1 = scatterplot(first_moment[1, 1, 1, 1:100])
scatterplot!(scp1, sras[end][1, 1, 1:100], color = :cyan)

scp2 = scatterplot(second_moment[1, 1, 1, 1:100])
scatterplot!(scp2, sras_2[end][1, 1, 1:100], color = :cyan)

scp3 = lineplot(25 * third_moment[1, 1, 1, 1:100])
lineplot!(scp3, 25 * sras_3[end][1, 1, 1:100], color = :cyan)

scp3 = lineplot(25 * third_moment[4, 1, 1, 1:100])
lineplot!(scp3, 25 * sras_3[end][4, 1, 1:100], color = :cyan)

scp3 = lineplot(25 * third_moment[8, 1, 1, 1:100])
lineplot!(scp3, 25 * sras_3[end][8, 1, 1:100], color = :cyan)

scp4 = scatterplot(fourth_moment[1, 1, 1, 1:100])
scatterplot!(scp4, sras_4[end][1, 1, 1:100], color = :cyan)

scp5 = scatterplot(fifth_moment[1, 1, 1, 1:100])
scatterplot!(scp5, sras_5[end][1, 1, 1:100], color = :cyan)

# print error for first pixels and 1:100 
for i in eachindex(sras)
    println("Error for first pixel and 1:100")
    println(5^0 * mean(abs.(sras[i][1, 1, 1:100] - first_moment[1, 1, 1, 1:100])))
    println(5^1 * mean(abs.(sras_2[i][1, 1, 1:100] - second_moment[1, 1, 1, 1:100])))
    println(5^2 * mean(abs.(sras_3[i][1, 1, 1:100] - third_moment[1, 1, 1, 1:100])))
    println(5^3 * mean(abs.(sras_4[i][1, 1, 1:100] - fourth_moment[1, 1, 1, 1:100])))
    println(5^4 * mean(abs.(sras_5[i][1, 1, 1:100] - fifth_moment[1, 1, 1, 1:100])))
    println("--------------------------------")
end




function hack(score_response_array, moment_at_zero)
    N, N, endindex = size(score_response_array)
    hack_response = zeros(N^2, N^2, endindex);
    A = zeros(N^2, N^2);
    @info "Constructing Hack Matrix"
    for k in ProgressBar(1:endindex)
        for i in 1:N^2
            ii = (i-1) % N + 1
            jj = (i-1) ÷ N + 1
            @inbounds hack_response[:, i, k] .= circshift(score_response_array[:, :, k], (ii-1, jj-1))[:]
            @inbounds A[:, i] .= circshift(moment_at_zero[:, :], (ii-1, jj-1))[:]
        end
    end
    @info "Inverting Hack Matrix"
    C⁻¹ = inv(hack_response[:, :, 1]) * A
    # hack_score_response = [(hack_response[:, :, i] * C⁻¹ + C⁻¹ * hack_response[:, :, i])/2  for i in 1:endindex]
    @info "Constructing Hack Response"
    return [hack_response[:, :, i] * C⁻¹ for i in ProgressBar(1:endindex)]
end


scp3 = lineplot(25 * third_moment[4, 1, 1, 1:100])
lineplot!(scp3, 25 * sras_3[end][4, 1, 1:100], color = :cyan)

# For first moment 
hrsa = hack(sras[end], first_moment[:, :, 1, 1])
generative_response_1 = zeros(N, N, length(hrsa))
for i in 1:N, j in 1:N
    generative_response_1[i, j, :] .= [reshape(hrsa[k][:, 1], (N, N))[i, j] for k in eachindex(hrsa)]
end

# For third moment
hrsa = hack(sras_3[end], third_moment[:, :, 1, 1])
generative_response_3 = zeros(N, N, length(hrsa))
for i in 1:N, j in 1:N
    generative_response_3[i, j, :] .= [reshape(hrsa[k][:, 1], (N, N))[i, j] for k in eachindex(hrsa)]
end

# For fifth moment
hrsa = hack(sras_5[end], fifth_moment[:, :, 1, 1])
generative_response_5 = zeros(N, N, length(hrsa))
for i in 1:N, j in 1:N
    generative_response_5[i, j, :] .= [reshape(hrsa[k][:, 1], (N, N))[i, j] for k in eachindex(hrsa)]
end


hfile = h5open("/orcd/data/raffaele/001/sandre/Repositories/ResponseFunction/CliMAgen.jl/"*"generative_responses.hdf5", "w")
hfile["score_response_1"] = sras[end]
hfile["score_response_2"] = sras_2[end]
hfile["score_response_3"] = sras_3[end]
hfile["score_response_4"] = sras_4[end]
hfile["score_response_5"] = sras_5[end]

hfile["dynamical_response_1"] = first_moment[:, :, 1, :]

hfile["hack_response_1"] = generative_response_1
hfile["hack_response_3"] = generative_response_3
hfile["hack_response_5"] = generative_response_5
close(hfile)