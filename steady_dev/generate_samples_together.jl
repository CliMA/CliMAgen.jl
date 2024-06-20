


#=
fixed_model = false
capacity = false
include("generate_samples_kat_1.jl")
=#

#=
fixed_model = false
capacity = true
include("generate_samples_kat_1.jl")
=#

#=
fixed_model = true
capacity = false
include("generate_samples_kat_1.jl")
=#


include("generate_samples_kat_2.jl")
for i in 1:4
    main(i)
end