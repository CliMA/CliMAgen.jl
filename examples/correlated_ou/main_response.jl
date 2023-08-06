include("GenerateTrajectories.jl")
include("Responses.jl")

using Statistics
using Plots
using HDF5, Random, ProgressBars

using Main.GenerateTrajectories: trajectory, regularization
using Main.Responses: response_lin, response_num, response_score

N = 8
beta = 0.1
gamma = 10
sigma_start = 1
Dt = 0.01
t_therm = 1000

data = trajectory([rand() for i in 1:N^2],N,100*t_therm,12345,sigma_start,Dt,1,beta,gamma)[:,t_therm+1:end]
std_data = std(data)

alpha = 5*std_data
sigma = sigma_start/alpha
t = 1000000

data = trajectory([rand()*sigma for i in 1:N^2],N,t+t_therm,12345,sigma,Dt,alpha,beta,gamma)[:,t_therm+1:end]

x = copy(regularization(data))

file_path = "data/data_$(beta)_$(gamma)_$(sigma_start).hdf5"
hfile = h5open(file_path, "w") 
hfile["timeseries"] = reshape(x,(N,N,1,t))
close(hfile)

plot(x[1,1:100:end])
##
tau = 100
n_ens = 1000
eps = 0.01

file_response_lin = "data/response_lin_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_response_num = "data/response_num_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_response_score = "data/response_score_$(beta)_$(gamma)_$(sigma_start).hdf5"

response_lin(tau, file_path, file_response_lin)
response_num(file_response_num,tau,n_ens,eps,alpha,beta,gamma,Dt,sigma,t_therm,N)
#response_score(tau, file_path, file_response_score)

##
hfile = h5open(file_response_lin) 
res_lin_mean = read(hfile["response_lin_mean"])
res_lin = read(hfile["response_lin"])
close(hfile)

hfile = h5open(file_response_num) 
res_num = read(hfile["response_num"])
error = read(hfile["error"])
close(hfile)

# hfile = h5open(file_response_score) 
# res_score_mean = read(hfile["response_score_mean"])
# res_score = read(hfile["response_score"])
# close(hfile)
##
pl = plot(
    plot([.-res_num[1,:].+error[1,:]./sqrt(n_ens) .-res_num[1,:].-error[1,:]./sqrt(n_ens) res_lin_mean[1,:]],label=["numerics" "" "linear app"],color=[:red :red :blue],title="1 -> 1",xlabel="time",ylabel="response"),
    plot([.-res_num[2,:].+error[2,:]./sqrt(n_ens) .-res_num[2,:].-error[2,:]./sqrt(n_ens) res_lin_mean[2,:]],label=["" "" ""],color=[:red :red :blue],title="1 -> 2",xlabel="time",ylabel="response"),
    plot([.-res_num[3,:].+error[3,:]./sqrt(n_ens) .-res_num[3,:].-error[3,:]./sqrt(n_ens) res_lin_mean[3,:]],label=["" "" ""],color=[:red :red :blue],title="1 -> 3",xlabel="time",ylabel="response"),
    plot([.-res_num[4,:].+error[4,:]./sqrt(n_ens) .-res_num[4,:].-error[4,:]./sqrt(n_ens) res_lin_mean[4,:]],label=["" "" ""],color=[:red :red :blue],title="1 -> 4",xlabel="time",ylabel="response"),
    layout=(2,2),
    size=(1000,1000),
    plot_title="beta = $beta, gamma = $gamma, sigma = $sigma_start"
)

display(pl)