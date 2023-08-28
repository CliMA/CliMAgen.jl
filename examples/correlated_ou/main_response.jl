using Revise
include("GetData.jl")
include("GenerateTrajectories.jl")
include("Responses.jl")
include("ScoreGenMod.jl")
# true is not the true response
using Statistics
using Plots
using HDF5, Random, ProgressBars

using Main.GenerateTrajectories: trajectory, regularization
using Main.Responses: response_lin, response_num, response_true, response_score
using Main.ScoreGenMod: generate_score

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
t = 300000

data = trajectory([rand()*sigma for i in 1:N^2],N,t+t_therm,12345,sigma,Dt,alpha,beta,gamma)[:,t_therm+1:end]

x = copy(regularization(data))

file_path = "correlated_ou/data/data_$(beta)_$(gamma)_$(sigma_start).hdf5"
hfile = h5open(file_path, "w") 
hfile["timeseries"] = reshape(x,(N,N,t))
close(hfile)

pl = plot(x[1,1:100:end])
display(pl)

@info "generating score"
generate_score(beta,gamma,sigma_start)

tau = 1000
n_ens = 1000
eps = 0.01
file_x = "correlated_ou/data/data_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_sc = "correlated_ou/data/scores_$(beta)_$(gamma)_$(sigma_start).hdf5"

file_response_lin = "correlated_ou/data/response_lin_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_response_true = "correlated_ou/data/response_true_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_response_num = "correlated_ou/data/response_num_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_response_score = "correlated_ou/data/response_score_$(beta)_$(gamma)_$(sigma_start).hdf5"

@info "linear response"
response_lin(tau, file_x, file_response_lin)
@info "true response"
response_true(tau, file_x, file_response_true,alpha,beta,gamma,sigma)
@info "numerical response"
response_num(file_response_num,tau,n_ens,eps,alpha,beta,gamma,Dt,sigma,t_therm,N)
@info "generative model response"
response_score(tau, file_sc, file_response_score)

##

# beta = 0.1
# gamma = 10
# sigma_start = 1

file_response_lin = "correlated_ou/data/response_lin_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_response_true = "correlated_ou/data/response_true_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_response_num = "correlated_ou/data/response_num_$(beta)_$(gamma)_$(sigma_start).hdf5"
file_response_score = "correlated_ou/data/response_score_$(beta)_$(gamma)_$(sigma_start).hdf5"

hfile = h5open(file_response_lin) 
res_lin_mean = read(hfile["response_lin_mean"])
res_lin = read(hfile["response_lin"])
close(hfile)

hfile = h5open(file_response_true) 
res_true_mean = read(hfile["response_true_mean"])
res_true = read(hfile["response_true"])
close(hfile)

hfile = h5open(file_response_num) 
res_num = read(hfile["response_num"])
error = read(hfile["error"])
close(hfile)

hfile = h5open(file_response_score) 
res_score_mean = read(hfile["response_score_mean"])
res_score = read(hfile["response_score"])
close(hfile)
##
tau = 1000
res_score2 = zeros(N^2,N^2,tau)
for i in 1:tau
    res_score2[:,:,i] = inv(res_score[:,:,1])*res_score[:,:,i]
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
    plot_title="beta = $beta, gamma = $gamma, sigma = $sigma_start"
)

display(pl)
savefig("correlated_ou/figures/fig_$(beta)_$(gamma)_$(sigma_start).png")
