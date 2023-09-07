include("GenerateTrajectories.jl")
include("Responses.jl")
include("GetData.jl")
include("ScoreGenMod.jl")

using Main.Responses: response_lin, response_num, response_true, response_score
using Main.ScoreGenMod: generate_score


using Statistics, StatsBase
using Plots
using HDF5, Random, ProgressBars,TOML

using Main.GenerateTrajectories: trajectory2, trajectory, regularization, regularization2

toml_dict = TOML.parsefile("correlated_ou/data/trj.toml")

alpha = toml_dict["param_group"]["alpha"]
beta = toml_dict["param_group"]["beta"]
gamma = toml_dict["param_group"]["gamma"]
sigma = toml_dict["param_group"]["sigma_start"]
n_std = toml_dict["param_group"]["n_std"]
t_start = toml_dict["param_group"]["t_start"]
res = toml_dict["param_group"]["res"]

N = 8
Dt = 0.02
t_therm = 1000

data = trajectory2([rand()*sigma for i in 1:N^2],N,t_start+t_therm,12345,sigma,Dt,alpha,beta,gamma,res)[:,1:end]
x = copy(regularization2(data,n_std))

plot(x[1,1:100:10000])
##
t = 100000
file_path = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
hfile = h5open(file_path, "r")
x = read(hfile["timeseries"])
x = reshape(x,(N*N,size(x)[3]))
close(hfile)

pl = plot(x[1,1:10:10000])
display(pl)

##
cum_x = zeros(10)
for i in 1:10
    cum_x[i] = cumulant(reshape(x,(64*size(x)[2])),i)
end

scatter(cum_x)
##
stephist(reshape(x,(64*size(x)[2])),normalize=:pdf)

##
file_path = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
hfile = h5open(file_path, "w") 
hfile["timeseries"] = reshape(x,(N,N,size(x)[2]))
close(hfile)
##
tau = 100
n_ens = 2000
eps = 0.01
file_x = "correlated_ou/data/data_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
file_sc = "correlated_ou/data/scores_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"

file_response_lin = "correlated_ou/data/response_lin_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
file_response_num = "correlated_ou/data/response_num_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
file_response_score = "correlated_ou/data/response_score_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"

response_lin(tau, file_x, file_response_lin)
response_num(file_response_num,10*tau,n_ens,eps,alpha,beta,gamma,Dt,sigma,t_therm,N)
response_score(tau, file_sc, file_response_score)
##
tau = 100
# beta = 0.1
# gamma = 20
# sigma_start = 2

file_response_lin = "correlated_ou/data/response_lin_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
file_response_num = "correlated_ou/data/response_num_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
file_response_score = "correlated_ou/data/response_score_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"

hfile = h5open(file_response_lin) 
res_lin_mean = read(hfile["response_lin_mean"])[:,1:tau]
res_lin = read(hfile["response_lin"])
close(hfile)

hfile = h5open(file_response_num) 
res_num = read(hfile["response_num"])[:,1:10:Int(tau*10)]
error = read(hfile["error"])[:,1:10:Int(10*tau)]
close(hfile)

hfile = h5open(file_response_score) 
res_score_mean = read(hfile["response_score_mean"])[:,1:tau]
res_score = read(hfile["response_score"])
close(hfile)

res_score2 = zeros(N^2,N^2,tau)
for i in 1:tau
    res_score2[:,:,i] = res_score[:,:,i]*inv(res_score[:,:,1])
end

distances = [0:(5-1)...]
res_score_mean2 = zeros(length(distances),tau)
for d in distances
    for i in 1:N^2 - distances[d+1]
        res_score_mean2[d+1,:] .+= res_score2[i,i+distances[d+1],:]
    end
    res_score_mean2[d+1,:] ./= (N^2 - distances[d+1])
end

pl = plot(
    plot([.-res_num[1,:].+error[1,:]./sqrt(n_ens) .-res_num[1,:].-error[1,:]./sqrt(n_ens) res_lin_mean[1,:] res_score_mean2[1,:]],label=["numerics" "" "linear app" "score"],color=[:red :red :blue :black],title="1 -> 1",xlabel="time",ylabel="response", linewidth=3),
    plot([.-res_num[2,:].+error[2,:]./sqrt(n_ens) .-res_num[2,:].-error[2,:]./sqrt(n_ens) res_lin_mean[2,:] res_score_mean2[2,:]],label=["" "" "" ""],color=[:red :red :blue :black],title="1 -> 2",xlabel="time",ylabel="response", linewidth=3),
    plot([.-res_num[3,:].+error[3,:]./sqrt(n_ens) .-res_num[3,:].-error[3,:]./sqrt(n_ens) res_lin_mean[3,:] res_score_mean2[3,:]],label=["" "" "" ""],color=[:red :red :blue :black],title="1 -> 3",xlabel="time",ylabel="response", linewidth=3),
    plot([.-res_num[4,:].+error[4,:]./sqrt(n_ens) .-res_num[4,:].-error[4,:]./sqrt(n_ens) res_lin_mean[4,:] res_score_mean2[4,:]],label=["" "" "" ""],color=[:red :red :blue :black],title="1 -> 4",xlabel="time",ylabel="response", linewidth=3),
    layout=(2,2),
    size=(1000,1000),
    plot_title="beta = $beta, gamma = $gamma, sigma = $sigma"
)

display(pl)
savefig("correlated_ou/figures/responses.png")

# plot(.-res_num[3,:])
# plot!(res_lin_mean[3,1:1:50])

##
# generate_score(alpha,beta,gamma,sigma; res=1)
# include("analysis2.jl")
# include("print_score.jl")
##