include("GetData.jl")

using BSON
using CUDA
using Flux
using Images
using ProgressMeter
using Plots
using Random
using Statistics
using TOML
using Main.GetData: get_data
using HDF5

using CliMAgen

# function Euler_Maruyama_sampler_linear(model::CliMAgen.AbstractDiffusionModel,
#     init_x::A,
#     time_steps,
#     Δt,
#     invC0;
#     c=nothing,
#     forward = false,
#     )::A where {A}
# x = mean_x = init_x

# @showprogress "Euler-Maruyama Sampling Linear" for time_step in time_steps
# batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
# g = CliMAgen.diffusion(model, batch_time_step)
# x_cpu = cpu(x)
# score = device(reshape(invC0*reshape(x_cpu, (size(x_cpu)[1]^2,size(x_cpu)[4])), (size(x_cpu)[1],size(x_cpu)[1],1,size(x_cpu)[4])))    
# mean_x = x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt
# x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
# end
# return x
# end

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function run_analysis(params, invC0; FT=Float32, logger=nothing)
toml_dict = TOML.parsefile("correlated_ou/data/trj.toml")

alpha = toml_dict["param_group"]["alpha"]
beta = toml_dict["param_group"]["beta"]
gamma = toml_dict["param_group"]["gamma"]
sigma = toml_dict["param_group"]["sigma_start"]
# read experiment parameters from file
savedir = params.experiment.savedir
rngseed = params.experiment.rngseed
nogpu = params.experiment.nogpu

batchsize = params.data.batchsize
fraction = params.data.fraction
inchannels = params.model.noised_channels
nsamples = params.sampling.nsamples
nsteps = params.sampling.nsteps
tilesize_sampling = params.sampling.tilesize
sampler = params.sampling.sampler

# set up rng
rngseed > 0 && Random.seed!(rngseed)

# set up device
if !nogpu && CUDA.has_cuda()
    device = Flux.gpu
    @info "Sampling on GPU"
else
    device = Flux.cpu
    @info "Sampling on CPU"
end
f_path = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
f_variable = "timeseries"
# set up dataset
dataloaders,_ = get_data(
    f_path, f_variable,batchsize;
    f = fraction,
    FT=Float32,
    rng=Random.GLOBAL_RNG
)

xtrain = first(dataloaders)

checkpoint_path = joinpath(savedir, "checkpoint_$(alpha)_$(beta)_$(gamma)_$(sigma).bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)
    
    # sample from the trained model
    time_steps, Δt, init_x = setup_sampler(
        model,
        device,
        tilesize_sampling,
        inchannels;
        num_images=nsamples,
        num_steps=nsteps,
    )
    if sampler == "euler"
        samples = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
        #samples_linear = Euler_Maruyama_sampler_linear(model, init_x, time_steps, Δt, invC0)
    elseif sampler == "pc"
        samples = predictor_corrector_sampler(model, init_x, time_steps, Δt)
    end
    samples = cpu(samples) 
    
    # spatial_mean_plot(xtrain, samples, savedir, "spatial_mean_distribution.png", logger=logger)
    # qq_plot(xtrain, samples, savedir, "qq_plot.png", logger=logger)
    # spectrum_plot(xtrain, samples, savedir, "mean_spectra.png", logger=logger)

    # # create plots with nimages images of sampled data and training data
    # # for ch in 1:inchannels
    # #     heatmap_grid(samples[:, :, [ch], 1:nimages], ch, savedir, "$(sampler)_images_$(ch).png")
    # #     heatmap_grid(xtrain[:, :, [ch], 1:nimages], ch, savedir, "train_images_$(ch).png")
    # # end
    
    # loss_plot(savedir, "losses.png"; xlog = false, ylog = true)    
    # #samples_linear = cpu(samples_linear)
    return xtrain, samples  
end

experiment_toml="correlated_ou/Experiment.toml"
FT = Float32

toml_dict = TOML.parsefile("correlated_ou/data/trj.toml")

alpha = toml_dict["param_group"]["alpha"]
beta = toml_dict["param_group"]["beta"]
gamma = toml_dict["param_group"]["gamma"]
sigma = toml_dict["param_group"]["sigma_start"]
file_path = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
hfile = h5open(file_path, "r") 
x = read(hfile["timeseries"])
close(hfile)
xt = transpose(reshape(x,(size(x)[1]^2,size(x)[3])))
invC0 = inv(cov(xt))

# read experiment parameters from file
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
logger = nothing
xtrain, samples = run_analysis(params, invC0; FT=FT, logger=logger);

##
cum_x = zeros(10)
cum_samples = zeros(10)
for i in 1:10
    cum_x[i] = cumulant(reshape(x,(64*size(x)[3])),i)
    cum_samples[i] = cumulant(reshape(samples,(64*size(samples)[4])),i)
end

scatter(cum_x,label="Data",xlabel="Cumulants")
scatter!(cum_samples, label="Gen")
savefig("correlated_ou/figures/cumulants.png")
##
stephist(reshape(x,(64*size(x)[3])),normalize=:pdf,label="Data",xlims=(-0.95,0.95),ylims=(0,5))
stephist!(reshape(samples,(64*size(samples)[4])),normalize=:pdf, label="Gen",xlims=(-0.95,0.95),ylims=(0,5))
savefig("correlated_ou/figures/pdfs.png")
# ##

##


# beta = 0.1
# gamma = 20
# sigma = 2
# # read experiment parameters from file
# savedir = params.experiment.savedir
# rngseed = params.experiment.rngseed
# nogpu = params.experiment.nogpu

# batchsize = params.data.batchsize
# resolution = params.data.resolution
# fraction = params.data.fraction
# standard_scaling = params.data.standard_scaling
# preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")

# sigma_min::FT = params.model.sigma_min
# sigma_max::FT = params.model.sigma_max
# inchannels = params.model.noised_channels
# shift_input = params.model.shift_input
# shift_output = params.model.shift_output
# mean_bypass = params.model.mean_bypass
# scale_mean_bypass = params.model.scale_mean_bypass
# gnorm = params.model.gnorm
# proj_kernelsize = params.model.proj_kernelsize
# outer_kernelsize = params.model.outer_kernelsize
# middle_kernelsize = params.model.middle_kernelsize
# inner_kernelsize = params.model.inner_kernelsize

# nsamples = params.sampling.nsamples
# nimages = params.sampling.nimages
# nsteps = params.sampling.nsteps
# tilesize_sampling = params.sampling.tilesize
# sampler = params.sampling.sampler

# nwarmup = params.optimizer.nwarmup
# gradnorm::FT = params.optimizer.gradnorm
# learning_rate::FT = params.optimizer.learning_rate
# beta_1::FT = params.optimizer.beta_1
# beta_2::FT = params.optimizer.beta_2
# epsilon::FT = params.optimizer.epsilon
# ema_rate::FT = params.optimizer.ema_rate

# nepochs = params.training.nepochs
# freq_chckpt = params.training.freq_chckpt

# # set up rng
# rngseed > 0 && Random.seed!(rngseed)

# # set up device
# if !nogpu && CUDA.has_cuda()
#     device = Flux.gpu
#     @info "Sampling on GPU"
# else
#     device = Flux.cpu
#     @info "Sampling on CPU"
# end
# f_path = "correlated_ou/data/data_$(beta)_$(gamma)_$(sigma).hdf5"
# f_variable = "timeseries"
# # set up dataset
# dataloaders,_ = get_data(
#     f_path, f_variable,batchsize;
#     f = fraction,
#     FT=Float32,
#     rng=Random.GLOBAL_RNG
# )

# xtrain = first(dataloaders)

# checkpoint_path = joinpath(savedir, "checkpoint_$(beta)_$(gamma)_$(sigma).bson")
#     BSON.@load checkpoint_path model model_smooth opt opt_smooth
#     model = device(model)
    
#     # sample from the trained model
#     time_steps, Δt, init_x = setup_sampler(
#         model,
#         device,
#         tilesize_sampling,
#         inchannels;
#         num_images=nsamples,
#         num_steps=nsteps,
#     )

#     Δt
# ##
# x = mean_x = init_x
# xx = init_xx = reshape(cpu(x),(64,size(x)[4]))[:,1]
# plot(xx)
# ##

# nn = 10
# @showprogress "Euler-Maruyama Sampling Linear" for time_step in 
# g = CliMAgen.diffusion(model, time_step)
# Δt /= 0.001
# A = .-g.*(invC0)*Δt
# xx = xx .+ A*xx .+ sqrt(Δt) .* g .* randn!(similar(xx))
# end
# plt = plot(xx)
# ##
# using LinearAlgebra
# v = eigen(invC0).values
# sort(v)
# surface(exp(-invC0 .* 1))
# ##
# @showprogress "Euler-Maruyama Sampling Linear" for time_step in time_steps
# batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
# g = CliMAgen.diffusion(model, batch_time_step)
# if forward
# x = x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
# else
# score = invC0*x
# mean_x = x .+ CliMAgen.expand_dims(g, 3) .^ 2 .* score .* Δt
# x = mean_x .+ sqrt(Δt) .* CliMAgen.expand_dims(g, 3) .* randn!(similar(x))
# end
# end
