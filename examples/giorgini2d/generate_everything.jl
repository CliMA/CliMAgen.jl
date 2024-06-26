# @info "Generating Data"
# include("generate_data.jl")
experiment_toml = "ExperimentNavier.toml"
model_toml = "ModelNavier.toml"

@info "Training Score Function"
include("training.jl")
main(; experiment_toml, model_toml) 
@info "Analysis"
include("analysis.jl")
main(; experiment_toml, model_toml) 
@info "Generating Score Response"
include("generate_score_response_allen_cahn.jl")
