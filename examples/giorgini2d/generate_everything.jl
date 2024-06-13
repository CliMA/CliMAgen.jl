# @info "Generating Data"
# include("generate_data.jl")
@info "Training Score Function"
include("training.jl")
main() 
@info "Analysis"
include("analysis.jl")
main() 
@info "Generating Score Response"
experiment_toml = "ExperimentLinear.toml"
model_toml = "ModelLinear.toml"
include("generate_score_response_allen_cahn.jl")
