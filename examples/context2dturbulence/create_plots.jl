using DelimitedFiles
using CliMAgen
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_analysis.jl"))

function main(; experiment_toml="Experiment.toml")
    FT = Float32
    # read experiment parameters from file
    params = TOML.parsefile(experiment_toml)
    params = CliMAgen.dict2nt(params)
    savedir = params.experiment.savedir
    gen_filenames = [joinpath(savedir, "gen_statistics_ch1.csv"),joinpath(savedir, "gen_statistics_ch2.csv")]
    train_filenames = [joinpath(savedir, "train_statistics_ch1.csv"),joinpath(savedir, "train_statistics_ch2.csv")]
    for ch in [1,2]
        # means, κ2, κ3, κ4, spectra
        gen_stats = readdlm(gen_filenames[ch], ',')
        train_stats = readdlm(train_filenames[ch], ',')
        # make plots
    end
end

        
