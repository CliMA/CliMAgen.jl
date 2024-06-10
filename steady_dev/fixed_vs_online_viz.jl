using HDF5, CairoMakie 
batch_scale = 4
hfile = h5open("losses_fixed_data.hdf5", "r")
fixed_generalization_loss = read(hfile["losses_3"]) * batch_scale# read(hfile["losses_2"])
fixed_test_loss = read(hfile["losses"]) * batch_scale
close(hfile)

hfile = h5open("losses_online.hdf5", "r")
# yes the order got reversed >.>
online_loss = read(hfile["losses"])
online_test_loss = read(hfile["losses_2"]) * batch_scale
close(hfile)

hfile = h5open("losses_online_more.hdf5", "r")
# yes the order got reversed >.>
online_loss_2 = read(hfile["losses_3"]) * 4 # for the new batch size
online_loss_3 = read(hfile["losses_4"]) * 4 # for the new batch size
close(hfile)

hfile = h5open("losses_online_capacity.hdf5", "r")
# online_capicity_loss = read(hfile["losses"])
online_capacity_test_lost = read(hfile["losses_2"]) * batch_scale
online_capacity_loss = read(hfile["losses_3"]) * batch_scale
online_capacity_test_lost_3 = read(hfile["losses_4"]) * batch_scale
close(hfile)


hfile = h5open("losses_online_rotations.hdf5", "r")
online_loss_rotation = read(hfile["losses"])
online_loss_rotation_test = read(hfile["losses_2"]) * batch_scale
close(hfile)

hfile = h5open("losses_online_rotation_start.hdf5", "r")
online_loss_rotation_start = read(hfile["losses"])
online_loss_rotation_start_test = read(hfile["losses_2"]) * batch_scale
# online_loss_rotation_start_2 = read(hfile["losses_3"]) * batch_scale
# online_loss_rotation_start_3 = read(hfile["losses_4"]) * batch_scale
close(hfile)


hfile = h5open("losses_online_fixed_start.hdf5", "r")
online_loss_fixed_start = read(hfile["losses"])
online_loss_fixed_start_test = read(hfile["losses_2"]) * batch_scale
close(hfile)

xmax = 400 # 110
scale = 1000
steps_online = collect(eachindex(online_loss)) .* 500 ./ scale
steps_fixed = collect(eachindex(fixed_generalization_loss)) .* 500 ./scale
steps_capacity = collect(eachindex(online_capacity_loss)) .* 500 ./scale
steps_rotation_start = collect(eachindex(online_loss_rotation_start)) .* 500 ./scale
fig = Figure(resolution = (1350, 450))
ax = Axis(fig[1, 1]; title = "Training Loss", xlabel ="$scale steps", ylabel = "loss")
lines!(ax, steps_fixed, fixed_test_loss, color = (:red, 0.5), label = "fixed data")
lines!(ax, steps_online, online_test_loss, color = (:blue, 0.5), label = "online training")
lines!(ax, steps_capacity, online_capacity_test_lost, color = (:orange, 0.5), label = "capacity")
# lines!(ax, steps_online, online_loss_rotation_test, color = (:purple, 0.5), label = "rotation")
axislegend(ax, position = :rt)
xlims!(ax, 0, xmax)
# xlims!(ax, 0 ./ scale, steps_online[end])
# ylims!(ax, 0.001, 0.005) # 0.018)
ylims!(ax, 0.0015, 0.03) 

ax = Axis(fig[1, 2]; title = "Generalization Loss", xlabel ="$scale steps", ylabel = "loss")
lines!(ax, steps_fixed, fixed_generalization_loss, color = (:red, 0.5), label = "fixed data")
lines!(ax, steps_online, online_loss_2, color = (:blue, 0.5), label = "online training")
# scatter!(ax, steps_online, online_loss_2, color = (:yellow, 0.5), markersize = 3, label = "online training (new loss)")
# scatter!(ax, steps_online, online_loss_3, color = (:SteelBlue, 0.5), markersize = 3, label = "online training (new loss)")
lines!(ax, steps_capacity, online_capacity_loss, color = (:orange, 0.5), label = "capacity")
# lines!(ax, steps_online, online_loss_rotation, color = (:purple, 0.5), label = "rotation")
# lines!(ax, steps_online, online_loss_capacity, color = (:orange, 0.5), label = "online training: higher capacity")
axislegend(ax, position = :ct)
xlims!(ax, 0, xmax)
ylims!(ax, 0.0015, 0.03) # 0.018)


ax = Axis(fig[1, 3]; title = "Generalization Loss: Strategies", xlabel ="$scale steps", ylabel = "loss")
steps_fixed_start = collect(eachindex(online_loss_fixed_start)) .* 500 ./scale
lines!(ax, steps_fixed_start, online_loss_fixed_start_test, color = (:red, 0.5), label = "fixed data start")
lines!(ax, steps_rotation_start, online_loss_rotation_start_test, color = (:purple, 0.5), label = "rotation start")
axislegend(ax, position = :rt)
xlims!(ax, 0, xmax)
ylims!(ax, 0.0015, 0.03)


save("losses_fixed_vs_online.png", fig)

##
fig = Figure(resolution = (1350, 450))
ax = Axis(fig[1, 1]; title = "Training Loss", xlabel ="1000 steps", ylabel = "loss")
lines!(ax, steps_online, online_loss_2, color = (:blue, 0.5), label = "online training 1")
lines!(ax, steps_online, online_loss_3, color = (:blue, 0.5), label = "online training 2")
axislegend(ax, position = :rt)

save("losses_online_new.png", fig)