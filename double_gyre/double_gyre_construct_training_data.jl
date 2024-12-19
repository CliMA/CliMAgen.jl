using HDF5, Statistics
data_directory = "/orcd/data/raffaele/001/sandre/DoubleGyreAnalysisData/"
data_directory_training = "/orcd/data/raffaele/001/sandre/DoubleGyreTrainingData/"

function coarse_grained(field, factor)
    N = size(field)[1]
    new_field = zeros(N, N)
    NN = N ÷ factor
    for i in 1:NN 
        for j in 1:NN 
            is = (i-1)*factor+1:i*factor
            js = (j-1)*factor+1:j*factor
            new_field[is, js] .= mean(field[is, js])
        end
    end
    return new_field
end

M = 128
casevar = 5
@info "opening data"
hfile = h5open(data_directory * "training_baroclinic_double_gyre_$(M)_$casevar.hdf5", "r")

zs = read(hfile["z"])
eta = read(hfile["eta"])
v = read(hfile["v"])

close(hfile)
@info "Creating DataSet 1: eta_to_v_at_z1"
level_index = 1
hfile = h5open(data_directory_training * "eta_to_v_at_z$(level_index)_$(M)_$(casevar).hdf5", "w")

eta_mean = mean(eta)
eta_2std = 2 * std(eta)
eta_rescaled = (eta .- eta_mean) ./ eta_2std

v_field = v[:, :, level_index:level_index, :]
v_mean = mean(v_field)
v_2std = 2 * std(v_field)
v_rescaled = (v_field .- v_mean) ./ v_2std

field = zeros(M, M, 2, size(eta)[4])
field[:, :, 1:1, :] .= v_rescaled
field[:, :, 2:2, :] .= eta_rescaled

hfile["eta"] = eta_rescaled
hfile["v"] = v_rescaled
hfile["field"] = field
hfile["eta_mean"] = eta_mean
hfile["eta_2std"] = eta_2std
hfile["v_mean"] = v_mean
hfile["v_2std"] = v_2std
hfile["zlevel"] = zs[level_index]

coarse_grained_field = copy(field)
for k in 1:5
    for i in 1:size(field, 4)
        coarse_grained_field[:, :, 2, i] .= coarse_grained(field[:, :, 2, i], 2^k)
    end
    hfile["field $(2^k)"] = coarse_grained_field
end
close(hfile)