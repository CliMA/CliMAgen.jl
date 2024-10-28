using LinearAlgebra
hfile = h5open("/nobackup1/sandre/OceananigansData/baroclinic_training_data_3.hdf5")
field = read(hfile["timeseries"])
close(hfile)

hfile = h5open("/nobackup1/sandre/OceananigansData/baroclinic_double_gyre_3.hdf5")
meanu = read(hfile["mean u"])
meanv = read(hfile["mean v"])
meanb = read(hfile["mean b"])
mean_eta = read(hfile["mean eta"])
stdu = read(hfile["std u"])
stdv = read(hfile["std v"])
stdb = read(hfile["std b"])
std_eta = read(hfile["std eta"])
close(hfile)

field[:, :, 1, :] .*= stdu[:, :, 1, :]
field[:, :, 2, :] .*= stdv[:, :, 1, :]
field[:, :, 3, :] .*= stdb[:, :, 1, :]
field[:, :, 4, :] .*= std_eta

field[:, :, 1, :] .+= meanu[:, :, 1, :]
field[:, :, 2, :] .+= meanv[:, :, 1, :]
field[:, :, 3, :] .+= meanb[:, :, 1, :]
field[:, :, 4, :] .+= mean_eta 

umean = mean(field[:, :, 1, :], dims=(1, 2, 3))
vmean = mean(field[:, :, 2, :], dims=(1, 2, 3))
bmean = mean(field[:, :, 3, :], dims=(1, 2, 3))
etamean = mean(field[:, :, 4, :], dims=(1, 2, 3))
ustd = std(field[:, :, 1, :], dims=(1, 2, 3)) * 2
vstd = std(field[:, :, 2, :], dims=(1, 2, 3)) * 2
bstd = std(field[:, :, 3, :], dims=(1, 2, 3)) * 2
etastd = std(field[:, :, 4, :], dims=(1, 2, 3)) * 2

field[:, :, 1, :] .-= umean
field[:, :, 2, :] .-= vmean
field[:, :, 3, :] .-= bmean
field[:, :, 4, :] .-= etamean

field[:, :, 1, :] ./= ustd
field[:, :, 2, :] ./= vstd
field[:, :, 3, :] ./= bstd
field[:, :, 4, :] ./= etastd

is = rand(1:size(field, 4),100)
js = rand(1:size(field, 4),100)
sigma_max = maximum([norm(field[:, :, 1:3, i] - field[:, :, 1:3, j]) for i in is, j in js]) 