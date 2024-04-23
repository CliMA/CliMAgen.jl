using TOML
using HDF5
using StatsBase
using CliMAgen
using CairoMakie
using JLD2

package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/conus404/preprocessing_utils.jl"))
include(joinpath(package_dir,"examples/conus404/plotting/utils.jl"))
include(joinpath(package_dir,"examples/conus404/plotting/pixel_plots.jl"))

function run_postprocessing(params; FT=Float32, subset_preprocess_params = "train")
    # unpack params
    savedir = params.experiment.savedir
    samples_file = "samples_smooth.hdf5"
    hdf5_path=joinpath(savedir, samples_file)
    fid = HDF5.h5open(hdf5_path, "r")
    samples = HDF5.read(fid["generated_samples"])
    ndata = size(samples)[end]
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$(standard_scaling)_$(subset_preprocess_params).jld2")
    scaling = JLD2.load_object(preprocess_params_file)
    # convert to real space
    samples_real_space = invert_preprocessing(samples, scaling)

    # read training and testing dataset
    xtrain, xtest = get_raw_data_conus404(; FT)
    nsample_train = size(xtrain)[end]
    id_arr = 1:1:nsample_train
    idx = StatsBase.sample(id_arr, ndata)
    xtrain = xtrain[:,:,:,idx]
    nsample_test = size(xtest)[end]
    id_arr = 1:1:nsample_test
    idx = StatsBase.sample(id_arr, ndata)
    xtest = xtest[:,:,:,idx]
    
    pixel_plots(xtrain, samples_real_space, ["training", "generated"], joinpath(savedir,"train_samples_$(subset_preprocess_params).png"))
    pixel_plots(xtest, samples_real_space, ["test", "generated"],joinpath(savedir,"test_samples_$(subset_preprocess_params).png"))
    pixel_plots(xtrain, xtest, ["training", "test"], joinpath(savedir,"train_test.png"))


    # Try also in preproc space
    xtrain_pp = apply_preprocessing(xtrain, scaling);
    pixel_plots(xtrain_pp, samples, ["training", "generated"], joinpath(savedir,"train_samples_$(subset_preprocess_params)_pp.png"))

    HDF5.close(fid)
end

function main(; experiment_toml="Experiment.toml", subset_preprocess_params = "train")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    run_postprocessing(params; FT=FT, subset_preprocess_params = subset_preprocess_params)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(;experiment_toml=ARGS[1], subset_preprocess_params = ARGS[2])
end
