using CairoMakie
using CliMADatasets

#snapshot_hr = Turbulence2DContext(; split=:train, resolution=512, wavenumber=:all, fraction=0.2)[:][:, :, :, 1]
#snapshot_lr = Turbulence2DContext(; split=:train, resolution=64, wavenumber=:all, fraction=0.2)[:][:, :, :, 1]

# fig
fig = Figure(resolution=(750, 750), fontsize=24)

ax = Axis(fig[1,1], ylabel="High resolution", title="Supersaturation", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, titlefont = :regular)
co = heatmap!(ax, snapshot_hr[:,:,1], colormap=Reverse(:blues))

ax = Axis(fig[1,2], title="Vorticity", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, titlefont = :regular)
heatmap!(ax, snapshot_hr[:,:,2], colormap=:bluesreds)

ax = Axis(fig[2,1], ylabel="Low resolution", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
co = heatmap!(ax, snapshot_lr[:,:,1], colormap=Reverse(:blues))

ax = Axis(fig[2,2], xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
heatmap!(ax, snapshot_lr[:,:,2], colormap=:bluesreds)

save("fig:snapshots.png", fig, px_per_unit = 2)