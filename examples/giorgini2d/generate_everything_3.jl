# @info "Generating Data"
# include("generate_data.jl")
experiment_toml = "ExperimentNavier3.toml"
model_toml = "ModelNavier3.toml"

@info "Training Score Function"
include("training.jl")
main(; experiment_toml, model_toml) 
@info "Analysis"
include("analysis.jl")
main(; experiment_toml, model_toml) 
@info "Generating Score Response"
include("generate_score_response_checkpoints.jl")
