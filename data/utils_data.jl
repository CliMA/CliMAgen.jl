using Pkg.Artifacts
using CliMAgen
using Flux
using HDF5
using MLUtils
using Random
using StatsBase

# datasets
include("datasets.jl")
include("dataloaders.jl")


function obtain_local_dataset_path(ds::ArtifactMetaData)
    data_dir = joinpath(pkgdir(CliMAgen), "data")
    return obtain_local_dataset_path(data_dir, ds.dataname, ds.url, ds.filename)
end

# Code modified from Artifacts.jl documentation
function obtain_local_dataset_path(dir, dataname, url, filename)
    artifact_toml = joinpath(dir, "Artifacts.toml")
    data_hash = artifact_hash(dataname, artifact_toml)

    # If the name was not bound, or the hash it was bound to does not exist, create it!
    if data_hash isa Nothing || !artifact_exists(data_hash)
        @info "Downloading data"
        # create_artifact() returns the content-hash of the artifact directory once we're finished creating it
        data_hash = create_artifact() do artifact_dir
            # We create the artifact by simply downloading into the new artifact directory
            download("$(url)", joinpath(artifact_dir, filename))
        end
        # Now bind that hash within our `Artifacts.toml`.  `force = true` means that if it already exists,
        # just overwrite with the new content-hash.  Unless the source files change, we do not expect
        # the content hash to change, so this should not cause unnecessary version control churn.
        bind_artifact!(artifact_toml, dataname, data_hash, force=true)
    end

    return joinpath(artifact_path(data_hash), filename)
end
 
