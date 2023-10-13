# This code computes the response functions numerically, using the linear approximation and the score function obtained from the generative model 

using Statistics
using Random
using LinearAlgebra
using ProgressBars
using TOML
using Plots
using StatsBase
using HDF5
using BSON
using CliMAgen
using JLD2
using CliMAgen: expand_dims, MeanSpatialScaling, StandardScaling, apply_preprocessing, invert_preprocessing

include("Tools.jl")

using Main.Tools: corr,regularization,s

toml_dict = TOML.parsefile("correlated_ou/trj_res.toml")

alpha = toml_dict["param_group"]["alpha"]
beta = toml_dict["param_group"]["beta"]
gamma = toml_dict["param_group"]["gamma"]
sigma = toml_dict["param_group"]["sigma"]
n_std = toml_dict["param_group"]["n_std"]
t = toml_dict["param_group"]["t"]
t_therm = toml_dict["param_group"]["t_therm"]
res = toml_dict["param_group"]["res"]
n_std = toml_dict["param_group"]["n_std"]
n_ens = toml_dict["param_group"]["n_ens"]

pfile = "output_preprocess_true_periodic_true_$(alpha)_$(beta)_$(gamma)_$(sigma)/preprocess.jld2"

N = 8
Dt = 0.02
seed = 12345
tau = 1000
n_dist = 5

function get_neighbors_avg(A, ind, D)
    N = size(A, 1)  # Assuming A is NxN
    idx = CartesianIndices(A)[(ind-1)*N+ind]  # Convert linear index to CartesianIndex
    i, j = idx[1], idx[2]
    indices = [(i+D, j), (i-D, j), (i, j+D), (i, j-D)]  # Neighbors above, below, right, and left

    # Apply periodic boundary conditions
    periodic_indices = [(mod(x[1]-1, N)+1, mod(x[2]-1, N)+1) for x in indices]

    # Compute average
    avg = mean([A[idx...] for idx in periodic_indices])
    return avg
end

function overall_avg(A, D)
    N = size(A, 1)
    avg_values = [get_neighbors_avg(A, ind, D) for ind in 1:N]
    return mean(avg_values)
end

function mean_response(response; n_dist=n_dist, tau=tau)
    mean_res = zeros(n_dist,tau)
    for i in 1:n_dist
        for j in 1:tau
            mean_res[i,j] = overall_avg(response[:,:,j],i-1)
        end
    end
    return mean_res
end

function scaling2D(x, preprocess_params_file; FT=Float32)
    scaling = JLD2.load_object(preprocess_params_file)
    return reshape(apply_preprocessing(reshape(x,(N,N,1,size(x)[2])), scaling), (N^2,size(x)[2]))
end

function rescaling2D(x, preprocess_params_file; FT=Float32)
    scaling = JLD2.load_object(preprocess_params_file)
    return reshape(invert_preprocessing(reshape(x,(N,N,1,size(x)[2])), scaling), (N^2,size(x)[2]))
end

function trajectory(X0,N,t,seed,sigma,Dt,alpha,beta,gamma; res=1, pfile=pfile)
    Random.seed!(seed)
    n_thr = Threads.nthreads()
    f(x) = s(x,N,alpha,beta,gamma)
    xOld = [X0[:,i] for i in 1:n_thr]
    xNew = [X0[:,i] for i in 1:n_thr]
    ΓL = LinearAlgebra.cholesky(corr(N)).L
    trj = zeros(N^2,Int(t/res),n_thr)
    trj[:,1,:] = X0
    count = zeros(Int,n_thr)
    for ens in 1:n_thr
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
    return trj
end

Random.seed!(seed)
f(x) = s(x,N,alpha,beta,gamma)
X0 = [(2*rand()-1) for i in 1:N^2]
xOld = [X0 for i in 1:n_ens]
xNew = [X0 for i in 1:n_ens]
ΓL = LinearAlgebra.cholesky(corr(N)).L
trj = zeros(N^2,Int((t+t_therm)/res),n_ens)
count = zeros(Int,n_ens)
Threads.@threads for ens in ProgressBar(1:n_ens)
    for i in 1:t+t_therm
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
trj = regularization(reshape(trj[:,Int(t_therm/res)+1:end,:],(N^2,Int(t/res)*n_ens)),n_std)
trj = scaling2D(trj,pfile)

trj_t = transpose(trj)
invC0 = inv(cov(trj_t))
score_lin = zeros(N^2,size(trj)[2])
for i in 1:size(trj)[2]
    score_lin[:,i] = invC0*trj[:,i]
end
score_lin_t = transpose(score_lin)
distances = [0:(n_dist-1)...]
responseL = zeros(N^2,N^2,tau)
Threads.@threads for i in ProgressBar(1:tau)
    responseL[:,:,i] = cov(trj_t[i:end,:],score_lin_t[1:end-i+1,:])
end

n_ens = 200
eps = 0.01
n_thr = Threads.nthreads()
responseN_ens = zeros(N^2,Int(tau/res),n_ens)
X0 = zeros(N^2,n_thr,n_ens)
X0eps = zeros(N^2,n_thr,n_ens)
for i in ProgressBar(1:n_ens)
    r1 = abs(rand(Int))
    X00 = 2*rand(N^2,n_thr).-1
    X0 = trajectory(X00,N,t_therm,r1,sigma,Dt,alpha,beta,gamma)[:,end,:]
    X0eps = copy(X0)
    X0eps[1,:] =  X0eps[1,:] .+ eps
    r2 = abs(rand(Int))
    t1 = trajectory(X0[:,:],N,tau,r2,sigma,Dt,alpha,beta,gamma)
    t2 = trajectory(X0eps[:,:],N,tau,r2,sigma,Dt,alpha,beta,gamma)
    responseN_ens[:,:,i] = mean((t1 .- t2)./eps,dims=3)
end
responseN = reshape(mean(responseN_ens, dims=3),(N^2,tau))
error = zeros(N^2,tau)
for i in 1:Int(tau/res)
    error[:,i] = std(responseN_ens[:,i,:], dims=2)
end

savedir = "output_preprocess_true_periodic_true_$(alpha)_$(beta)_$(gamma)_$(sigma)"
checkpoint_path = joinpath(savedir, "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth

times = [1:t...]  
t0 = 0.  

x = reshape(trj, (N,N,1,n_thr*t))
@time scores = Array(CliMAgen.score(model, Float32.(x), t0));

f_path = "data/scores_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5"
hfile = h5open(f_path,"w")
write(hfile,"scores",scores)
write(hfile,"x",x)
close(hfile)

# f_path = "data/scores_$(alpha)_$(beta)_$(gamma)_$(sigma)_3.hdf5"
# hfile = h5open(f_path,"r")
# scores = read(hfile,"scores")
# trj_scaled = read(hfile,"x")
# close(hfile)

x = copy(trj)
sc = reshape(scores,(N*N,n_thr*t))
xt = transpose(x)
sct = transpose(sc)
distances = [0:(n_dist-1)...]
responseS = zeros(N^2,N^2,tau)
for i in ProgressBar(1:tau)
    responseS[:,:,i] = cov(xt[i:end,:],sct[1:end-i+1,:])
end

responseS_norm = zeros(N^2,N^2,tau)
for i in 1:tau
    responseS_norm[:,:,i] = responseS[:,:,i]*inv(responseS[:,:,1])
end

responseL_mean = mean_response(responseL)
responseS_mean = mean_response(responseS_norm)

hfile = h5open("data/response_$(alpha)_$(beta)_$(gamma)_$(sigma).hdf5","w")  
write(hfile,"responseN",responseN)
write(hfile,"error",error)
write(hfile,"responseL",responseL)
write(hfile,"responseL_mean",responseL_mean)
write(hfile,"responseS",responseS_norm)
write(hfile,"responseS_mean",responseS_mean)
close(hfile)

pl = plot(
    plot([.-responseN[1,:].+error[1,:]./sqrt(n_thr*n_ens) .-responseN[1,:].-error[1,:]./sqrt(n_thr*n_ens) responseL_mean[1,:] responseS_mean[1,:]],label=["numerics" "" "linear app" "score"],color=[:red :red :blue :black],title="1 -> 1",xlabel="time",ylabel="response", linewidth=3),
    plot([.-responseN[2,:].+error[2,:]./sqrt(n_thr*n_ens) .-responseN[2,:].-error[2,:]./sqrt(n_thr*n_ens) responseL_mean[2,:] responseS_mean[2,:]],label=["" "" "" ""],color=[:red :red :blue :black],title="1 -> 2",xlabel="time",ylabel="response", linewidth=3),
    # plot([.-responseN[3,:].+error[3,:]./sqrt(n_thr*n_ens) .-responseN[3,:].-error[3,:]./sqrt(n_thr*n_ens) responseL_mean[3,:] responseS_norm_mean[3,:]],label=["" "" "" ""],color=[:red :red :blue :black],title="1 -> 3",xlabel="time",ylabel="response", linewidth=3),
    # plot([.-responseN[4,:].+error[4,:]./sqrt(n_thr*n_ens) .-responseN[4,:].-error[4,:]./sqrt(n_thr*n_ens) responseL_mean[4,:] responseS_norm_mean[4,:]],label=["" "" "" ""],color=[:red :red :blue :black],title="1 -> 4",xlabel="time",ylabel="response", linewidth=3),
    layout=(1,2),
    size=(1000,500),
    plot_title="beta = $beta, gamma = $gamma, sigma = $sigma"
)

display(pl)
savefig("output_preprocess_true_periodic_true_$(alpha)_$(beta)_$(gamma)_$(sigma)/responses.png")
