using CUDA
using Random
using TOML
using DelimitedFiles

using CliMAgen
using CliMAgen: dict2nt
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl")) # for data loading
include(joinpath(package_dir,"examples/utils_analysis.jl")) # for compute_sigma_max
function convert_to_symbol(string)
    if string == "strong"
        return :strong
    elseif string == "medium"
        return :medium
    elseif string == "weak"
        return :weak
    else
        @error("Nonlinearity must be weak, medium, or strong.")
    end
end
function main(;experiment_toml)
    FT = Float32

    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)

    # set up directory for saving checkpoints
    !ispath(params.experiment.savedir) && mkpath(params.experiment.savedir)
    savedir = params.experiment.savedir
    rngseed = params.experiment.rngseed
    nogpu = params.experiment.nogpu

    batchsize = params.data.batchsize
    resolution = params.data.resolution
    fraction = params.data.fraction
    nonlinearity = convert_to_symbol(params.data.nonlinearity) 
    standard_scaling = params.data.standard_scaling
    preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
    # set up dataset
    dataloaders = get_data_giorgini2d(batchsize, resolution, nonlinearity; f = fraction,
                                      FT=FT,
                                      rng=Random.GLOBAL_RNG,
                                      standard_scaling = standard_scaling,
                                      read = false,
                                      save = true,
                                      preprocess_params_file = preprocess_params_file)
    σ_max = reduce(max,map(compute_sigma_max, dataloaders[1]))
    @show(σ_max)
    rm(preprocess_params_file)
end
if abspath(PROGRAM_FILE) == @__FILE__
    main(experiment_toml = ARGS[1])
end
