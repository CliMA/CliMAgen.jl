using BSON
using CliMAgen
using HDF5

N = 8
toml_dict = TOML.parsefile("correlated_ou/data/trj.toml")

alpha = toml_dict["param_group"]["alpha"]
beta = toml_dict["param_group"]["beta"]
gamma = toml_dict["param_group"]["gamma"]
sigma = toml_dict["param_group"]["sigma_start"]

savedir = "output"
checkpoint_path = joinpath(savedir, "checkpoint_$(alpha)_$(beta)_$(gamma)_$(sigma).bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth

hfile = h5open("correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5")
data = read(hfile["timeseries"])[:,:,1:500000]
close(hfile) 

data = data[:,:,1:end]
times = [1:size(data)[3]...]  
t0 = 0.  
N = size(data)[1]
x = reshape(data[:,:,times], (N,N,1,length(times)))
@time scores = Array(CliMAgen.score(model, Float32.(x), t0));
scores = reshape(scores, (N,N,length(times)))

f_path = "correlated_ou/data/scores_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
hfile = h5open(f_path,"w")
write(hfile,"scores",scores)
write(hfile,"timeseries",data)
close(hfile)