using LinearAlgebra
@info "reading data"
hfile = h5open(data_directory_training * "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar).hdf5", "r")
field = read(hfile["field $factor"])
close(hfile)

is = rand(1:size(field, 4),100)
js = rand(1:size(field, 4),100)
nlast = size(field, 3) - 1
sigma_max = maximum([norm(field[:, :, 1:nlast, i] - field[:, :, 1:nlast, j]) for i in is, j in js])