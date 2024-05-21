using CairoMakie, HDF5 

hfile = h5open("steady_default_data.hdf5", "r")
timeseries = read(hfile["timeseries"])
μ = read(hfile, "shift")
σ = read(hfile, "scaling")
lon = read(hfile["lon"])
lat = read(hfile["lat"])
close(hfile)

physical_timeseries = σ .* timeseries .+ μ

ensemble_indices = round.(Int, collect(range(1, 1000, length = 16)))
M = floor(Int, sqrt(length(ensemble_indices)))

fig = Figure(resolution = (800, 800))
for (i,j) in enumerate(ensemble_indices)
    ii = (i-1)÷M  + 1
    jj = (i-1)%M  + 1
    ax = Axis(fig[ii, jj]; xlabel = "Longitude", ylabel = "Latitude")
    heatmap!(ax, lon, lat, physical_timeseries[:,:,1,j], colormap = :thermometer, colorrange = (215, 291))
end

save("temperature_heatmap.png", fig)


fig = Figure(resolution = (800, 800))
for (i,j) in enumerate(ensemble_indices)
    ii = (i-1)÷M + 1
    jj = (i-1)%M + 1
    ax = Axis(fig[ii, jj]; xlabel = "Longitude", ylabel = "Latitude")
    hist!(ax, physical_timeseries[:,:,1,j][:], normalization = :pdf)
    xlims!(ax, 210, 300)
end
save("temperature_pixel_histgrams.png", fig)


fig = Figure(resolution = (800, 800))
ax = Axis(fig[1,1]; xlabel = "Longitude", ylabel = "Latitude")
hist!(ax, physical_timeseries[:,:,1,1][:],   normalization = :pdf, color = (:blue, 0.5), bins = 100, label = "Early")
hist!(ax, physical_timeseries[:,:,1,end][:], normalization = :pdf, color = (:red, 0.5), bins = 100, label = "Late")
axislegend(ax, position = :lt)
ax = Axis(fig[1,2]; xlabel = "Longitude", ylabel = "Latitude")
hist!(ax, physical_timeseries[:,:,1,500][:],   normalization = :pdf, color = (:blue, 0.5), bins = 100, label = "Mid")
hist!(ax, physical_timeseries[:,:,1,end][:], normalization = :pdf, color = (:red, 0.5), bins = 100, label = "Late")
axislegend(ax, position = :lt)
drift = mean(physical_timeseries, dims = (1,2))[:]
ax = Axis(fig[2, 1]; xlabel = "Time", ylabel = "Temperature")
lines!(ax, drift, color = :black, label = "Drift")
save("temperature_pixel_histgrams_overlap.png", fig)