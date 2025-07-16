experiment_toml = "ExperimentNavier4.toml"
model_toml = "ModelNavier4.toml"


@info "Training Score Function"
include("training.jl")
main(; experiment_toml, model_toml) 
@info "Analysis"
include("analysis.jl")
main(; experiment_toml, model_toml) 
@info "Generating Score Response"
include("generate_score_response_checkpoints.jl")
