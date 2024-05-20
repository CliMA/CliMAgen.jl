using HDF5, CairoMakie 

hfile = h5open("losses_fixed_data.hdf5", "r")
fixed_generalization_loss = read(hfile["losses_2"])
close(hfile)

hfile = h5open("losses_online.hdf5", "r")
online_loss = read(hfile["losses"])[5:5:end] # 500 steps is 1 epoch
close(hfile)

hfile = h5open("losses_online_capacity.hdf5", "r")
online_loss_capacity = read(hfile["losses"])[5:5:end] # 500 steps is 1 epoch
close(hfile)

scale = 1000
steps_online = collect(eachindex(online_loss)) .* 500 ./ scale
steps_fixed = collect(eachindex(fixed_generalization_loss)) .* 500 ./scale
fig = Figure()
ax = Axis(fig[1, 1]; title = "losses", xlabel ="$scale steps", ylabel = "loss")
lines!(ax, steps_fixed, fixed_generalization_loss, color = (:red, 0.5), label = "fixed data")
lines!(ax, steps_online, online_loss, color = (:blue, 0.5), label = "online training")
lines!(ax, steps_online, online_loss_capacity, color = (:orange, 0.5), label = "online training: higher capacity")
axislegend(ax, position = :rt)
xlims!(ax, 0 ./ scale, steps_online[end])
ylims!(ax, 0.003, 0.018)

save("losses_fixed_vs_online.png", fig)