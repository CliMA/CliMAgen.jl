experiment_toml = "ExperimentLinear.toml"
model_toml = "ModelLinear.toml"

@info "Training Score Function"
include("training.jl")
main(;  model_toml, experiment_toml) 
@info "Analysis"
include("analysis.jl")
main(;  model_toml, experiment_toml) 
@info "Generating Score Response"
include("generate_score_response_allen_cahn.jl")