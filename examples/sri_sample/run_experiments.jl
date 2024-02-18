@info "Running experiments"
#=
@info "Experiment 1"
include("training_context.jl")
main(;model_toml="Model1.toml")
include("analysis_context.jl")
main(;model_toml="Model1.toml")

@info "Experiment 2"
include("training_context.jl")
main(;model_toml="Model2.toml")
include("analysis_context.jl")
main(;model_toml="Model2.toml")
=# 

@info "Experiment 3"
include("training_context.jl")
main(;model_toml="Model3.toml")
include("analysis_context.jl")
main(;model_toml="Model3.toml")

@info "Experiment 4"
include("training_context.jl")
main(;model_toml="Model4.toml")
include("analysis_context.jl")
main(;model_toml="Model4.toml")

@info "Experiment 5"
include("training_context.jl")
main(;model_toml="Model5.toml")
include("analysis_context.jl")
main(;model_toml="Model5.toml")

