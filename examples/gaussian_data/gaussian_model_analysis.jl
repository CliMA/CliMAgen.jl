using Flux
using CUDA
using cuDNN
using Dates
using Random
using TOML
using BSON 
using MLUtils
using DataLoaders
using Plots
using ProgressMeter
using Printf
using CliMAgen
using CliMAgen: dict2nt,load_model_and_optimizer

package_dir = pkgdir(CliMAgen)
include("../utils_etl.jl")
include("gaussian_data.jl")
function loss_plot(savepath::String, plotname::String; xlog::Bool=false, ylog::Bool=true)
    path = joinpath(savepath,plotname)
    filename = joinpath(savepath, "losses.txt")
    data = DelimitedFiles.readdlm(filename, ',', skipstart = 1)
    
    if size(data)[2] == 5
        plt1 = plot(left_margin = 20Plots.mm, ylabel = "Log10(Mean Loss)")
	plt2 = plot(bottom_margin = 10Plots.mm, left_margin = 20Plots.mm,xlabel = "Epoch", ylabel = "Log10(Spatial Loss)")
	plot!(plt1, data[:,1], data[:,2], label = "Train", linecolor = :black)
    	plot!(plt1, data[:,1], data[:,4], label = "Test", linecolor = :red)
    	plot!(plt2, data[:,1], data[:,3], label = "", linecolor = :black)
    	plot!(plt2, data[:,1], data[:,5], label = "", linecolor = :red)
    	if xlog
           plot!(plt1, xaxis=:log)
           plot!(plt2, xaxis=:log)
    	end
    	if ylog
           plot!(plt1, yaxis=:log)
           plot!(plt2, yaxis=:log)
        end
	plot(plt1, plt2, layout =(2,1))
	savefig(path)
    elseif size(data)[2] == 3
        plt1 = plot(left_margin = 20Plots.mm, ylabel = "Log10(Loss)")
	plot!(plt1, data[:,1], data[:,2], label = "Train", linecolor = :black)
    	plot!(plt1, data[:,1], data[:,3], label = "Test", linecolor = :red)
    	if xlog
           plot!(plt1, xaxis=:log)
    	end
    	if ylog
           plot!(plt1, yaxis=:log)
        end
	savefig(path)
    else
        @info "Loss CSV file has incorrect number of columns"
    end
end

function quick_analysis_plots(params; FT = FT)
    # unpack params
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu
    batchsize = params.data.batchsize
    inchannels = params.model.inchannels
    nsteps = params.sampling.nsteps
    nsamples = params.sampling.nsamples
    samples_file = params.sampling.samples_file
    tilesize = 16
    preprocess_params_file = joinpath(savedir, "preprocessing.jld2")
    dataloaders = get_data_gaussian(batchsize,preprocess_params_file;
                                    tilesize = 16,
                                    read =true,
                                    save=false,
                                    FT=FT
                                    )
    xtrain = cat([x for x in dataloaders[1]]..., dims=4);
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

    # set up model
    checkpoint_path = joinpath(savedir, "checkpoint.bson")
    BSON.@load checkpoint_path model model_smooth opt opt_smooth
    model = device(model)

    # sample from the trained model
    samples_per_batch = batchsize
    nbatch = div(nsamples, samples_per_batch)
    all_samples = zeros(FT, (tilesize, tilesize, inchannels,nbatch*samples_per_batch))
    samples = zeros(FT, (tilesize, tilesize, inchannels,samples_per_batch)) |> device
    for b in 1:nbatch
        time_steps, Δt, init_x = setup_sampler(
            model,
            device,
            tilesize,
            inchannels;
            num_images=samples_per_batch,
            num_steps=nsteps,
        )
        samples .= Euler_Maruyama_ld_sampler(model, init_x, time_steps, Δt, rng = MersenneTwister(b))
        all_samples[:,:,:,(b-1)*samples_per_batch+1:b*samples_per_batch] .= cpu(samples)
    end
    loss_plot(params.experiment.savedir, "losses.png")
    Plots.histogram(all_samples[:], label = "generated", title="Pixel distribution", xlabel = "Pixel value", norm = true)
    Plots.histogram!(xtrain[:], label = "training", norm = true)
    Plots.savefig(joinpath(savedir, "pixel_histogram.png"))
    generated_mean = mean(all_samples[:])
    generated_var = var(all_samples[:])
    train_mean = mean(xtrain[:])
    train_var = var(xtrain[:])
    N = prod(size(xtrain))
    standard_error_mean = sqrt(train_var/N)
    standard_error_var = sqrt(2)*train_var/sqrt(N)
    @printf "Means: Gen %.4f Train %.4f Uncertainty %.4f \n" generated_mean train_mean standard_error_mean
    @printf "Variance: Gen %.4f Train %.4f Uncertainty %.4f \n" generated_var train_var standard_error_var
end

function main(;experiment_toml = "Experiment_gaussian_sigmamax.toml")
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    quick_analysis_plots(params; FT = FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(;experiment_toml = ARGS[1])
end



