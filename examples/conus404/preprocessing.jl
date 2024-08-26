using TOML

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/conus404/preprocessing_utils.jl"))
include(joinpath(package_dir,"examples/utils_data.jl"))
"""
    compute_sigma_max(x)

   Returns σ_max for the dataset `x`, which
is assumed to be of size nx x ny x nc x n_obs.
"""
function compute_sigma_max(x)
    n_obs = size(x)[end]
    max_distance = 0
    for i in 1:n_obs
        for j in i+1:n_obs
            distance = sqrt(sum((x[:,:,:,i] .- x[:,:,:,j]).^2))
            max_distance = max(max_distance, distance)
        end
    end
    return max_distance
end

function run_preprocess(params; FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    batchsize = params.data.batchsize
    standard_scaling = params.data.standard_scaling
    low_pass = params.data.low_pass
    low_pass_k = params.data.low_pass_k
    fname_train = params.data.fname_train
    fname_test = params.data.fname_test
    precip_channel = params.data.precip_channel
    precip_floor::FT = params.data.precip_floor
    preprocess_params_file_train = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_train.jld2")
    preprocess_params_file_test = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_test.jld2")

    # set up dataset
    xtrain, xtest = get_raw_data_conus404(fname_train, fname_test, precip_channel; precip_floor = precip_floor, FT=FT)
    save_preprocessing_params(
        xtrain, preprocess_params_file_train; 
        standard_scaling=standard_scaling,
        low_pass=low_pass,
        low_pass_k=low_pass_k,
        FT=FT,
    )
    save_preprocessing_params(
        xtest, preprocess_params_file_test; 
        standard_scaling=standard_scaling,
        low_pass=low_pass,
        low_pass_k=low_pass_k,
        FT=FT,
    )
    # compute sigma max
    dataloaders = get_data_conus404(fname_train, fname_test, precip_channel, batchsize;
        precip_floor = precip_floor, FT=FT, preprocess_params_file=preprocess_params_file_train)
    sigma_max = reduce(max,map(compute_sigma_max, dataloaders[1]))
    @info sigma_max
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    !ispath(params.experiment.savedir) && mkpath(params.experiment.savedir)
    run_preprocess(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
