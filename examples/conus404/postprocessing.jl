using TOML
using HDF5
using StatsBase
using CliMAgen
using CairoMakie
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))
include(joinpath(package_dir,"examples/conus404/preprocessing_utils.jl"))
include(joinpath(package_dir,"examples/conus404/plotting/utils.jl"))
include(joinpath(package_dir,"examples/conus404/plotting/pixel_plots.jl"))

function postprocessing(params; FT=Float32)
    ndata = 100

    # unpack params
    savedir = params.experiment.savedir
    samplesdir = savedir
    samples_file = params.sampling.samples_file
    !ispath(samplesdir) && mkpath(samplesdir)
    hdf5_path=joinpath(samplesdir, samples_file)
    fid = HDF5.h5open(hdf5_path, "r")
    samples = HDF5.read(fid["generated_samples"])
    xsample = FT.(samples)

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
    
    pixel_plots(xtest, xsample)

    HDF5.close(fid)
end

function main(; experiment_toml="Experiment.toml")
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    postprocessing(params; FT=FT)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml=ARGS[1])
end
