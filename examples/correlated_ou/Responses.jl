include("GenerateTrajectories.jl")

using Statistics
using Plots
using HDF5, Random, ProgressBars

using Main.GenerateTrajectories: gradient_periodic, laplacian_periodic, trajectory, regularization, s

N = 8
alpha = 1
beta = 0.1
gamma = 10
Dt = 0.01
sigma1 = 1
t_therm = 1000
t1 = 100000

data = trajectory([rand()*sigma1 for i in 1:N^2],N,t1+t_therm,1,sigma1,Dt,alpha,beta,gamma)[:,t_therm+1:end]
std_data = std(data)

alpha = 4*std_data
sigma = sigma1 / (4*std_data)
t_therm = 1000
t = 100000

data = trajectory([rand()*sigma for i in 1:N^2],N,t+t_therm,1,sigma,Dt,alpha,beta,gamma)[:,t_therm+1:end]
println(std(data))

x = copy(regularization(data))
plot(x[1,1:100:end])
##
file_path = "data/data_$(beta)_$(sigma1)_$(gamma).hdf5"
hfile = h5open(file_path, "w") 
hfile["x"] = x[:,1:10:end]
close(hfile)
##
# file_path = "data/x4dataresponse.hdf5"
# hfile = h5open(file_path) 
# x_2D = read(hfile["timeseries"])
# s_2D = read(hfile["score"])
# x = reshape(x_2D,(N*N,size(x_2D)[3]*size(x_2D)[4]))[:,1:t]
# sc = reshape(s_2D,(N*N,size(x_2D)[3]*size(x_2D)[4]))[:,1:t]
# close(hfile)
# plot(x[1,1:100:end])
# ##
M = mean(x)
STD = std(x)
xt = transpose(x)
invC0 = inv(cov(xt))
#score_ML = .-sc
score_true = zeros(N^2,size(x)[2])
score_lin = zeros(N^2,size(x)[2])
for i in ProgressBar(1:size(x)[2])
    score_true[:,i] = .-s(x[:,i],N,alpha,beta,gamma)*2/sigma^2
    score_lin[:,i] = invC0*(x[:,i].-M)
end
score_true_t = transpose(score_true)
#score_ML_t = transpose(score_ML)
score_lin_t = transpose(score_lin)
# ##
# ind = rand(1:10000)
# plot(score_true[1:end,ind])
# plot!(score_lin[1:end,ind])
# #plot!(score_ML[:,ind])
# ##
tau = 500
n_ens = 1000
eps = 0.01
n_dist = 5
distances = [0:(n_dist-1)...]

response_lin = zeros(N^2,N^2,tau)
for i in ProgressBar(1:tau)
    response_lin[:,:,i] = cov(xt[i:end,:],score_lin_t[1:end-i+1,:])
end
response_lin_mean = zeros(length(distances),tau)
for d in distances
    for i in 1:N^2 - distances[d+1]
        response_lin_mean[d+1,:] .+= response_lin[i,i+distances[d+1],:]
    end
    response_lin_mean[d+1,:] ./= (N^2 - distances[d+1])
end

# response_true = zeros(N^2,N^2,tau)
# for i in ProgressBar(1:tau)
#     response_true[:,:,i] = cov(xt[i:end,:],score_true_t[1:end-i+1,:])
# end
# response_true_mean = zeros(length(distances),tau)
# for d in distances
#     for i in 1:N^2 - distances[d+1]
#         response_true_mean[d+1,:] .+= response_true[i,i+distances[d+1],:]
#     end
#     response_true_mean[d+1,:] ./= (N^2 - distances[d+1])
# end

# response_ML = zeros(N^2,N^2,tau)
# for i in ProgressBar(1:tau)
#     response_ML[:,:,i] = cov((xt[i:end,:].-M),score_ML_t[1:end-i+1,:])
# end
# response_ML_mean = zeros(length(distances),tau)
# for d in distances
#     for i in 1:N^2 - distances[d+1]
#         response_ML_mean[d+1,:] .+= response_ML[i,i+distances[d+1],:]
#     end
#     response_ML_mean[d+1,:] ./= (N^2 - distances[d+1])
# end
response_num_ens = zeros(n_dist,tau,n_ens)
X0 = zeros(N^2,n_ens)
X0eps = zeros(N^2,n_ens)

Threads.@threads for i in ProgressBar(1:n_ens)
    r1 = abs(rand(Int))
    X0[:,i] = trajectory([rand()*sigma for i in 1:N^2],N,t_therm,r1,sigma,Dt,alpha,beta,gamma)[:,end]
    X0eps[:,i] = copy(X0[:,i])
    X0eps[1,i] += eps
    r2 = abs(rand(Int))
    response_num_ens[:,:,i] = (trajectory(X0[:,i],N,tau,r2,sigma,Dt,alpha,beta,gamma) .- trajectory(X0eps[:,i],N,tau,r2,sigma,Dt,alpha,beta,gamma))[1:n_dist,:]./eps
end
response_num = mean(response_num_ens, dims=3)
#response_num_ens = 0.
##
error = zeros(n_dist,tau)
for i in 1:tau
    error[:,i] = std(response_num_ens[:,i,:], dims=2)
end

file_path = "data/responses_$(beta)_$(sigma1)_$(gamma).hdf5"
hfile = h5open(file_path, "w") 
hfile["response_lin_mean"] = response_lin_mean
#hfile["response_true_mean"] = response_true_mean
hfile["response_num"] = response_num
hfile["error"] = error
#hfile["response_ML"] = response_ML
close(hfile)
##
beta = 0.1
sigma1 = 1
gamma = 1
file_path = "data/responses_$(beta)_$(sigma1)_$(gamma).hdf5"
hfile = h5open(file_path) 
response_lin_mean = read(hfile["response_lin_mean"])
#response_true_mean = read(hfile["response_true_mean"])
response_num = read(hfile["response_num"])
#response_ML = read(hfile["response_ML"])
close(hfile)
##
error = zeros(n_dist,tau)
for i in 1:tau
    error[:,i] = std(response_num_ens[:,i,:], dims=2)
end

ind = 2
pl = plot()
plot!(pl,.-response_num[ind,:] .+ error[ind,:]/sqrt(n_ens),label="numerics",color=:red)
plot!(pl,.-response_num[ind,:] .- error[ind,:]/sqrt(n_ens),label="numerics",color=:red)
plot!(pl,response_lin_mean[ind,:],label="linear",color=:blue)
display(pl)

#plot!(response_ML_mean[ind,:].-response_ML_mean[ind,1],label="ML")
