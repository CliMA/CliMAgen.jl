index_1 = [59, 90]
index_2 = [130, 46]
index_3 = [140, 47]
total_samples_physical = (total_samples .* physical_sigma) .+ physical_mu
rfield = (reshape(oldfield, 192, 96, 2, 251, 45) .* physical_sigma) .+ physical_mu


rfield = (reshape(oldfield, (192, 96, 1, 251, 45)) .* physical_sigma) .+ physical_mu

function saveplot(location_indices, ntotal, total_samples_physical, rfield, filename)
    fig  = Figure(resolution = (400 * length(location_indices), 800))
    binsize = 30
    for (i, location_index) in enumerate(location_indices)
        ax = Axis(fig[1, i]; title = "ai ($location_index)")
        hist!(ax, Array(total_samples_physical[location_index[1],location_index[2],1,1:ntotal÷2])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
        hist!(ax, Array(total_samples_physical[location_index[1],location_index[2],1,(ntotal÷2+1):end])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
        xlims!(ax, 295, 310)
    end
end
n = 2
ax = Axis(fig[2, 1]; title = "data ($index_1)")

hist!(ax, Array(rfield[index_1[1],index_1[2],1, 35-n:35+n, :])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(rfield[index_1[1],index_1[2],1, end-5:end, :])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax, 295, 310)
ax = Axis(fig[2, 2]; title = "data ($index_2)")
hist!(ax, Array(rfield[index_2[1],index_2[2],1, 35-n:35+n, :])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(rfield[index_2[1],index_2[2],1, end-5:end, :])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax,  296, 303)
ax = Axis(fig[2, 3]; title = "data ($index_3)")
hist!(ax, Array(rfield[index_3[1],index_3[2],1,35-n:35+n, :])[:], bins = binsize, color = (:blue, 0.5), normalization = :pdf)
hist!(ax, Array(rfield[index_3[1],index_3[2],1, end-5:end, :])[:], bins = binsize, color = (:orange, 0.5), normalization = :pdf)
# xlims!(ax, 260, 280)
save("samples_hist_pr_temp.png", fig)