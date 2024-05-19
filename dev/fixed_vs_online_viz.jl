using HDF5, CairoMakie 


hfile = h5open("losses_fixed_data.hdf5", "r")
fixed_generalization_loss = read(hfile["losses_2"])
close(hfile)

#=
fig = Figure()
ax = Axis(fig[1, 1]; title = "losses", xlabel ="epoch", ylabel = "loss")
lines!(ax, losses2, color = :red, label = "loss fixed data")
scatter!(ax, losses_online[5:5:end], color = (:blue, 0.25), label = "loss online training")
axislegend(ax, position = :rt)
save("losses_fixed_vs_online.png", fig)
=#