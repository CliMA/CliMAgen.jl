using TOML

using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/conus404/preprocessing_utils.jl"))
include(joinpath(package_dir,"examples/utils_data.jl"))

function run_preprocess(params; FT=Float32)
    # unpack params
    savedir = params.experiment.savedir
    standard_scaling = params.data.standard_scaling
    low_pass = params.data.low_pass
    low_pass_k = params.data.low_pass_k
    preprocess_params_file_train = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_train.jld2")
    preprocess_params_file_test = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_test.jld2")

    # set up dataset
    xtrain, xtest = get_raw_data_conus404(FT=FT)
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
