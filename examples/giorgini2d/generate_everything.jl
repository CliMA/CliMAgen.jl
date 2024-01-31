# @info "Generating Data"
# include("generate_data.jl")
@info "Training Score Function"
include("training.jl")
main() 
@info "Generating Score Response"
include("generate_score_response.jl")

@info "Generating Linear Response"
include("generate_linear_response.jl")
@info "Generating Numerical Response"
include("generate_numerical_response.jl")