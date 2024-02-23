#=
@info "Experiment 0"
include("training.jl")
main(;model_toml="Model0.toml", experiment_toml="Experiment0.toml")
include("analysis.jl")
main(;model_toml="Model0.toml", experiment_toml="Experiment0.toml")

@info "Experiment 1"
include("training_context.jl")
main(;model_toml="Model1.toml", experiment_toml="Experiment1.toml")
include("analysis_context.jl")
main(;model_toml="Model1.toml", experiment_toml="Experiment1.toml")

@info "Experiment 2"
include("training_context.jl")
main(;model_toml="Model2.toml", experiment_toml="Experiment2.toml")
include("analysis_context.jl")
main(;model_toml="Model2.toml", experiment_toml="Experiment2.toml")

@info "Experiment 3"
include("training_context.jl")
main(;model_toml="Model3.toml", experiment_toml="Experiment3.toml")
include("analysis_context.jl")
main(;model_toml="Model3.toml", experiment_toml="Experiment3.toml")
=#

@info "Experiment 4"
include("training_context.jl")
main(;model_toml="Model4.toml", experiment_toml="Experiment4.toml")
include("analysis_context.jl")
main(;model_toml="Model4.toml", experiment_toml="Experiment4.toml")


