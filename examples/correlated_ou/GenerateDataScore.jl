# This code generates the data used to train the score-based generative model

using Statistics
using Random
using LinearAlgebra
using ProgressBars
using TOML
using Plots
using StatsBase
using HDF5

include("Tools.jl")
using Main.Tools: corr,regularization,s

toml_dict = TOML.parsefile("correlated_ou/trj_score.toml")

alpha = toml_dict["param_group"]["alpha"]
beta = toml_dict["param_group"]["beta"]
gamma = toml_dict["param_group"]["gamma"]
sigma = toml_dict["param_group"]["sigma"]
n_std = toml_dict["param_group"]["n_std"]
t = toml_dict["param_group"]["t"]
res = toml_dict["param_group"]["res"]
n_std = toml_dict["param_group"]["n_std"]
n_ens = toml_dict["param_group"]["n_ens"]

N = 8
Dt = 0.02
seed = 12345

Random.seed!(seed)
f(x) = s(x,N,alpha,beta,gamma)
X0 = [(2*rand()-1) for i in 1:N^2]
xOld = [X0 for i in 1:n_ens]
xNew = [X0 for i in 1:n_ens]
ΓL = LinearAlgebra.cholesky(corr(N)).L
trj = zeros(N^2,Int(t/res),n_ens)
count = zeros(Int,n_ens)
Threads.@threads for ens in ProgressBar(1:n_ens)
    for i in 1:t
        r = randn(N^2)
        k1 = f(xOld[ens])
        y = xOld[ens] + Dt * k1 * 0.5
        k2 = f(y)
        y = xOld[ens] + Dt * k2 * 0.5
        k3 = f(y)
        y = xOld[ens] + Dt * k3
        k4 = f(y)
        r_corr = copy(r)
        mul!(r_corr,ΓL,r)
        xNew[ens] += Dt / 6. * (k1 + 2 * k2 + 2 * k3 + k4) + sqrt(Dt) .* (sigma .* r_corr)
        if i % res == 0
            count[ens] += 1
            trj[:,count[ens],ens] = xNew[ens]
        end
        xOld[ens] = xNew[ens]
    end
end
trj = regularization(reshape(trj,(N^2,Int(t/res)*n_ens)),n_std)

file_path = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
hfile = h5open(file_path, "w") 
hfile["timeseries"] = reshape(trj,(N,N,size(trj)[2]))
close(hfile)
##
pl1 = plot(trj[1,1:1000])

lags = [0:100...]
acf = autocor(trj[1,:],lags)
pl2 = plot(acf)

pl3 = stephist(reshape(trj,(N^2*size(trj)[2])),normalize=:pdf)

cum = zeros(10)
for i in 1:10
    cum[i] = cumulant(reshape(trj,(64*size(trj)[2])),i)
end
pl4 = scatter(cum)

display(pl1)
display(pl2)
display(pl3)
display(pl4)

