using LinearAlgebra
@info "reading data"
hfile = h5open(data_directory_training * "eta_to_v_at_z$(level_index)_$(M)_$(casevar).hdf5", "r")
field = read(hfile["field $factor"])
close(hfile)

is = rand(1:size(field, 4),100)
js = rand(1:size(field, 4),100)
sigma_max = maximum([norm(field[:, :, 1:2, i] - field[:, :, 1:2, j]) for i in is, j in js]) 