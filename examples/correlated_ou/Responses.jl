include("GenerateTrajectories.jl")

using Statistics
using Plots
using HDF5, Random, ProgressBars

using Main.GenerateTrajectories: gradient_periodic, laplacian_periodic, trajectory, regularization, s

N = 8
alpha = 3#5
beta = 2
gamma = 2*π
Dt = 0.01
sigma = 1/alpha
t_therm = 1000
t = 100000

data = trajectory([rand()*sigma for i in 1:N^2],N,t+t_therm,1,sigma,Dt,alpha,beta,gamma)[:,t_therm+1:end]
x = copy(regularization(data))
plot(data[1,1:100:end])
##
file_path = "data/xdata_sin2.hdf5"
hfile = h5open(file_path, "w") 
hfile["x"] = x
close(hfile)
##
file_path = "data/x4dataresponse.hdf5"
hfile = h5open(file_path) 
x_2D = read(hfile["timeseries"])
s_2D = read(hfile["score"])
x = reshape(x_2D,(N*N,size(x_2D)[3]*size(x_2D)[4]))[:,1:t]
sc = reshape(s_2D,(N*N,size(x_2D)[3]*size(x_2D)[4]))[:,1:t]
close(hfile)
plot(x[1,1:100:end])
##
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
##
ind = rand(1:10000)
plot(score_true[1:end,ind])
plot!(score_lin[1:end,ind])
#plot!(score_ML[:,ind])
##
tau = 200
n_ens = 10000
eps = 0.01
distances = [0,1,2,3]

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

response_true = zeros(N^2,N^2,tau)
for i in ProgressBar(1:tau)
    response_true[:,:,i] = cov(xt[i:end,:],score_true_t[1:end-i+1,:])
end
response_true_mean = zeros(length(distances),tau)
for d in distances
    for i in 1:N^2 - distances[d+1]
        response_true_mean[d+1,:] .+= response_true[i,i+distances[d+1],:]
    end
    response_true_mean[d+1,:] ./= (N^2 - distances[d+1])
end

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

response_num = zeros(N^2,tau)
for i in ProgressBar(1:n_ens)
    X0 = trajectory([rand()*sigma for i in 1:N^2],N,t_therm,i,sigma,Dt,alpha,beta,gamma)[:,end] 
    X0eps = copy(X0)
    X0eps[1] += eps
    response_num[:,:] .+= (trajectory(X0,N,tau,i,sigma,Dt,alpha,beta,gamma) .- trajectory(X0eps,N,tau,i,sigma,Dt,alpha,beta,gamma))
end
response_num .*= (1/n_ens/eps)
##
file_path = "data/responses.hdf5"
hfile = h5open(file_path, "w") 
hfile["response_lin_mean"] = response_lin_mean
hfile["response_true_mean"] = response_true_mean
hfile["response_num"] = response_num
#hfile["response_ML"] = response_ML
close(hfile)
##
file_path = "data/responses.hdf5"
hfile = h5open(file_path) 
response_lin_mean = read(hfile["response_lin_mean"])
response_true_mean = read(hfile["response_true_mean"])
response_num = read(hfile["response_num"])
response_ML = read(hfile["response_ML"])
close(hfile)
##
ind = 2
plot(response_true_mean[ind,:],label="true")
plot(.-response_num[ind,:],label="numerics")
plot!(response_lin_mean[ind,:],label="linear")
#plot!(response_ML_mean[ind,:].-response_ML_mean[ind,1],label="ML")