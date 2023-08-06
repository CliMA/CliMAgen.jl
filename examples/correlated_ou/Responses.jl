module Responses

include("GenerateTrajectories.jl")

using Statistics
using Plots
using HDF5, Random, ProgressBars

using Main.GenerateTrajectories: trajectory, regularization

export response_lin, response_num, response_score

function response_lin(tau, file_in, file_out; n_dist=5)
    hfile = h5open(file_in) 
    x_2D = read(hfile["timeseries"])
    N = size(x_2D)[1]
    t = size(x_2D)[4]
    x = reshape(x_2D,(N*N,t))
    close(hfile)
    xt = transpose(x)
    invC0 = inv(cov(xt))
    score_lin = zeros(N^2,size(x)[2])
    for i in ProgressBar(1:size(x)[2])
        score_lin[:,i] = invC0*x[:,i]
    end
    score_lin_t = transpose(score_lin)
    distances = [0:(n_dist-1)...]
    response = zeros(N^2,N^2,tau)
    for i in ProgressBar(1:tau)
        response[:,:,i] = cov(xt[i:end,:],score_lin_t[1:end-i+1,:])
    end
    response_mean = zeros(length(distances),tau)
    for d in distances
        for i in 1:N^2 - distances[d+1]
            response_mean[d+1,:] .+= response[i,i+distances[d+1],:]
        end
        response_mean[d+1,:] ./= (N^2 - distances[d+1])
    end
    hfile = h5open(file_out, "w") 
    hfile["response_lin"] = response
    hfile["response_lin_mean"] = response_mean
    close(hfile)
end

function response_num(file_out,tau,n_ens,eps,alpha,beta,gamma,Dt,sigma,t_therm,N; n_dist=5)
    response_ens = zeros(n_dist,tau,n_ens)
    X0 = zeros(N^2,n_ens)
    X0eps = zeros(N^2,n_ens)
    Threads.@threads for i in ProgressBar(1:n_ens)
        r1 = abs(rand(Int))
        X0[:,i] = regularization(trajectory([rand()*sigma for i in 1:N^2],N,t_therm,r1,sigma,Dt,alpha,beta,gamma))[:,end]
        X0eps[:,i] = copy(X0[:,i])
        X0eps[1,i] += eps
        r2 = abs(rand(Int))
        response_ens[:,:,i] = (regularization(trajectory(X0[:,i],N,tau,r2,sigma,Dt,alpha,beta,gamma)) .- regularization(trajectory(X0eps[:,i],N,tau,r2,sigma,Dt,alpha,beta,gamma)))[1:n_dist,:]./eps
    end
    response = mean(response_ens, dims=3)
    error = zeros(n_dist,tau)
    for i in 1:tau
        error[:,i] = std(response_ens[:,i,:], dims=2)
    end
    hfile = h5open(file_out, "w") 
    hfile["response_num"] = response
    hfile["error"] = error
    close(hfile)
end

function response_score(tau, file_in, file_out; n_dist=5)
    hfile = h5open(file_in) 
    x_2D = read(hfile["timeseries"])
    sc_2D = read(hfile["score"])
    N = size(x_2D)[1]
    t = size(x_2D)[4]
    x = reshape(x_2D,(N*N,t))
    sc = reshape(sc_2D,(N*N,t))
    close(hfile)
    xt = transpose(x)
    sct = transpose(sc)
    distances = [0:(n_dist-1)...]
    response = zeros(N^2,N^2,tau)
    for i in ProgressBar(1:tau)
        response[:,:,i] = cov(xt[i:end,:],sct[1:end-i+1,:])
    end
    response_mean = zeros(length(distances),tau)
    for d in distances
        for i in 1:N^2 - distances[d+1]
            response_mean[d+1,:] .+= response[i,i+distances[d+1],:]
        end
        response_mean[d+1,:] ./= (N^2 - distances[d+1])
    end
    hfile = h5open(file_out, "w") 
    hfile["response_score"] = response
    hfile["response_score_mean"] = response_mean
    close(hfile)
end
end
