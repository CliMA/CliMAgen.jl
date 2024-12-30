using HDF5, Statistics, ProgressBars

M = 128
kmax = round(Int, log2(M))
casevar = 5

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

@info "opening data"
hfile = h5open(data_directory * "training_baroclinic_double_gyre_$(M)_$(casevar)_complement.hdf5", "r")

zs = read(hfile["z"])
@info "reading eta"
eta = read(hfile["eta"])
@info "reading u"
u = read(hfile["u"])
@info "reading v"
v = read(hfile["v"])
@info "reading w"
w = read(hfile["w"])
@info "reading b"
b = read(hfile["b"])
close(hfile)

@info "Creating DataSets: eta_to_uvwb_at_zs"
for level_index in ProgressBar(eachindex(zs))
    hfile = h5open(data_directory_training * "eta_to_uvwb_at_z$(level_index)_$(M)_$(casevar)_complement.hdf5", "w")

    eta_mean = mean(eta)
    eta_2std = 2 * std(eta)
    eta_rescaled = (eta .- eta_mean) ./ eta_2std

    u_field = u[:, :, level_index:level_index, :]
    u_mean = mean(u_field)
    u_2std = 2 * std(u_field)
    u_rescaled = (u_field .- u_mean) ./ u_2std

    v_field = v[:, :, level_index:level_index, :]
    v_mean = mean(v_field)
    v_2std = 2 * std(v_field)
    v_rescaled = (v_field .- v_mean) ./ v_2std

    w_field = w[:, :, level_index:level_index, :]
    w_mean = mean(w_field)
    w_2std = 2 * std(w_field)
    w_rescaled = (w_field .- w_mean) ./ w_2std

    b_field = b[:, :, level_index:level_index, :]
    b_mean = mean(b_field)
    b_2std = 2 * std(b_field)
    b_rescaled = (b_field .- b_mean) ./ b_2std

    field = zeros(M, M, 5, size(eta)[4])
    field[:, :, 1:1, :] .= u_rescaled
    field[:, :, 2:2, :] .= v_rescaled
    field[:, :, 3:3, :] .= w_rescaled
    field[:, :, 4:4, :] .= b_rescaled
    field[:, :, 5:5, :] .= eta_rescaled

    hfile["eta"] = eta_rescaled
    hfile["u"] = u_rescaled
    hfile["v"] = v_rescaled
    hfile["w"] = w_rescaled
    hfile["b"] = b_rescaled
    hfile["field"] = field
    hfile["eta_mean"] = eta_mean
    hfile["eta_2std"] = eta_2std
    hfile["u_mean"] = u_mean
    hfile["u_2std"] = u_2std
    hfile["v_mean"] = v_mean
    hfile["v_2std"] = v_2std
    hfile["w_mean"] = w_mean
    hfile["w_2std"] = w_2std
    hfile["b_mean"] = b_mean
    hfile["b_2std"] = b_2std
    hfile["zlevel"] = zs[level_index]

    coarse_grained_field = copy(field)
    for k in 1:kmax
        for i in 1:size(field, 4)
            coarse_grained_field[:, :, end, i] .= coarse_grained(field[:, :, end, i], 2^k)
        end
        hfile["field $(2^k)"] = coarse_grained_field
    end
    close(hfile)
end