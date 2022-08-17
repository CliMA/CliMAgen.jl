using Test
using Base.Filesystem: rm
using Downscaling
path_to_utils = joinpath(pkgdir(Downscaling), "examples/artifact_utils.jl")
include(path_to_utils)

@testset "Artifacts" begin
    url= "https://caltech.box.com/shared/static/3gp1h6h18b1a3mm1uytooz9xazsxpo14.csv"
    filename = "test_artifact.csv"
    dataname = "test_artifact"
    
    
    data_info = ArtifactDataInformation(filename, dataname, url)
    @test data_info.url == url
    @test data_info.filename == filename
    @test data_info.dataname == dataname
    dir = joinpath(pkgdir(Downscaling),"test")
    artifact_file = joinpath(dir, "Artifacts.toml")
    
    # Make sure it downloads cleanly starting from scratch    
    local_dataset_path = obtain_local_dataset_path(dir, data_info.dataname, data_info.url, data_info.filename)
    @test read(joinpath(local_dataset_path, filename), String) == "1, 2, 3\n4, 5, 6"
    # make sure running again returns the same path
    @test obtain_local_dataset_path(dir, data_info.dataname, data_info.url, data_info.filename) == local_dataset_path

    # make sure when someone has the Artifact file but not the data, that it downloads
    rm(local_dataset_path; recursive = true)
    local_dataset_path = obtain_local_dataset_path(dir, data_info.dataname, data_info.url, data_info.filename)
    @test read(joinpath(local_dataset_path, filename), String) == "1, 2, 3\n4, 5, 6"

    # make sure if someone has the data but not the Artifact file, that it
    # is generated
    rm(artifact_file)
    local_dataset_path = obtain_local_dataset_path(dir, data_info.dataname, data_info.url, data_info.filename)
    @test read(artifact_file, String) == "[test_artifact]\ngit-tree-sha1 = \"cf6b2ffa0104dfde0658e7bad08f368dd3939999\"\n"

    # clean up so that the test always starts from scratch
    rm(local_dataset_path; recursive = true)
    rm(artifact_file)

end
