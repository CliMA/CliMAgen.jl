# This code computes the response functions numerically, using the linear approximation and the score function

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
using Distributions
package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/utils_data.jl"))

include("./utils.jl")
include("./rhs.jl")

FT = Float64

function convert_to_symbol(string)
    if string == "strong"
        return :strong
    elseif string == "medium"
        return :medium
    elseif string == "weak"
        return :weak
    else
        @error("Nonlinearity must be weak, medium, or strong.")
    end
end

experiment_toml="giorgini2d/Experiment_8_strong.toml"
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
savedir = "output_8x8_strong"
resolution = params.data.resolution
standard_scaling = params.data.standard_scaling
fraction = params.data.fraction
nonlinearity = convert_to_symbol(params.data.nonlinearity) 
preprocess_params_file = joinpath(savedir, "preprocessing_standard_scaling_$standard_scaling.jld2")
pfile = JLD2.load_object(preprocess_params_file)

N = 8
α = FT(0.3)
β = FT(0.5)
γ = FT(10)
σ = FT(2.0)
T = 500000
tau = 100
dt = FT(0.1)
dt_save = 10*dt
n_tau = Int(tau/dt_save)
n_T = Int(floor(T/dt_save))

function get_neighbors_avg(A, ind, D)
    N = size(A, 1)
    idx = CartesianIndices(A)[(ind-1)*N+ind]
    i, j = idx[1], idx[2]
    indices = [(i+D, j), (i-D, j), (i, j+D), (i, j-D)]
    periodic_indices = [(mod(x[1]-1, N)+1, mod(x[2]-1, N)+1) for x in indices]
    avg = mean([A[idx...] for idx in periodic_indices])
    return avg
end

function overall_avg(A, D)
    N = size(A, 1)
    avg_values = [get_neighbors_avg(A, ind, D) for ind in 1:N]
    return mean(avg_values)
end

function mean_response(response; n_dist=4, tau=n_tau)
    mean_res = zeros(n_dist,tau)
    for i in 1:n_dist
        for j in 1:tau
            mean_res[i,j] = overall_avg(response[:,:,j],i-1)
        end
    end
    return mean_res
end

function scaling2D(x; pfile=pfile, FT=Float32)
    return reshape(apply_preprocessing(reshape(x,(N,N,1,size(x)[2])), pfile), (N^2,size(x)[2]))
end

function rescaling2D(x; pfile=pfile, FT=Float32)
    return reshape(invert_preprocessing(x, pfile), (N^2,size(x)[4]))
end

Γ = FT.(reshape(zeros(N^4), (N^2,N^2)))
for i1 in 1:N
    for j1 in 1:N
        k1 = (j1-1)*N+i1
        for i2 in 1:N
            for j2 in 1:N
                k2 = (j2-1)*N+i2
                Γ[k1,k2] = FT(1/sqrt(min(abs(i1-i2),N-abs(i1-i2))^2 + min(abs(j1-j2),N-abs(j1-j2))^2+1))
            end
        end
    end
end
ΓL = cholesky(Γ).L
W = zeros(FT, N*N)
W_corr = similar(W)
Model = LudoDiffusionSDE(σ, α, β, γ, N, Periodic(), ΓL, W, W_corr)

function trajectory(u, tspan, dt, dt_save, seed; model = Model)
    Random.seed!(abs(seed))
    deterministic_tendency! = make_deterministic_tendency(model)
    stochastic_increment! = make_stochastic_increment(model)
    du = similar(u)
    nsteps = Int(floor((tspan[2]-tspan[1])/dt))
    n_steps_per_save = Int(round(dt_save/dt))
    savesteps = 0:n_steps_per_save:nsteps - n_steps_per_save
    solution = zeros(FT, (N^2, Int(nsteps/n_steps_per_save)))
    solution[:, 1] .= reshape(u, (N^2,))
    for i in 1:nsteps
        t = tspan[1]+dt*(i-1)
        Euler_Maruyama_step!(du, u, t, deterministic_tendency!, stochastic_increment!, dt)
        if i ∈ savesteps
            save_index = Int(i/n_steps_per_save)
            solution[:, save_index+1] .= reshape(u, (N^2,))
        end
    end
    return solution
end

tspan = FT.((0.0,T))
u0 = 2*rand(FT, N^2).-1

trj = trajectory(u0, tspan, dt, dt_save, rand(Int))
trj = scaling2D(trj)


dataloaders,_ = get_data_giorgini2d(4000, resolution, nonlinearity;
                                      f = fraction,
                                      FT=FT,
                                      rng=Random.GLOBAL_RNG,
                                      standard_scaling = standard_scaling,
                                      read = false,
                                      save = true,
                                      preprocess_params_file = preprocess_params_file)

dataf = first(dataloaders)
trj2 = reshape(dataf, (N^2,4000))

pl1 = plot(trj[1,1:1:1000],title="time series",xlabel="time",linewidth=3,label="")

lags = [0:5000...]
acf = autocor(trj[1,:],lags)
t_decorr = 0.
for i in eachindex(acf)
    if acf[i] < 0.1
        t_decorr = i*dt_save
        break
    end
end
pl2 = plot(acf[1:Int(floor(2*t_decorr/dt_save))],title="autocorrelation",xlabel="time",linewidth=3)

mu = mean(trj)
sig = std(trj)
dist = Normal(mu, sig)
xax = collect(mu-6*sig:0.01:mu+6sig)  
yax = map(v -> pdf(dist, v), xax)

pl3 = plot(xax, yax, label="normal distribution", linewidth=3, title="distribution of the time series", legend=:topleft)
pl3 = stephist!(reshape(trj,(N^2*size(trj)[2])),normalize=:pdf,label="histogram",linewidth=3)
pl3 = stephist!(reshape(trj2,(N^2*size(trj2)[2])),normalize=:pdf,label="histogram 2",linewidth=3)

cum = zeros(4)
for i in 1:4
    cum[i] = cumulant(reshape(trj,(N^2*size(trj)[2])),i)
end
pl4 = scatter(cum, label="cumulants", title="cumulants", xlabel="order")

pl = plot(pl1,pl2,pl3,pl4,layout=(2,2),size=(1000,700))
display(pl)

#savefig(pl, "trj_gamma_$γ.png")

trj_t = transpose(trj) .- mean(trj) 
invC0 = inv(cov(trj_t))
responseL = zeros(N^2,N^2,n_tau)
for i in ProgressBar(1:n_tau)
    responseL[:,:,i] = cov(trj_t[i:end,:],trj_t[1:end-i+1,:]) * invC0
end
responseL_mean = mean_response(responseL)

n_ens = 1000
eps = 0.01
responseN_ens = zeros(N^2,n_tau,n_ens)

for i in ProgressBar(1:n_ens)
    R1 = rand(Int)
    u0 = 2*rand(FT, N^2).-1
    tspan = FT.((0.0,2*t_decorr))
    X0 = trajectory(u0, tspan, dt, dt_save, R1)[:,end]
    X0eps = copy(X0)
    X0eps[1] += eps
    tspan = FT.((0.0,tau))
    R2 = rand(Int)
    t1 = trajectory(X0, tspan, dt, dt_save, R2)
    t2 = trajectory(X0eps, tspan, dt, dt_save, R2)
    responseN_ens[:,:,i] = (t2 .- t1)./eps
end
responseN = reshape(mean(responseN_ens, dims=3),(N^2,n_tau))
err = zeros(N^2,n_tau)
for i in 1:n_tau
    err[:,i] = std(responseN_ens[:,i,:], dims=2)
end

checkpoint_path = joinpath(savedir, "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth

res = 5
t = 50000
t0 = 0.  

pfile = joinpath(savedir, "preprocessing_standard_scaling_false.jld2")
x = reshape(trj[:,1:res:res*t], (N,N,1,t))
@time scores = Array(CliMAgen.score(model, Float32.(x), t0));

x = copy(trj[:,1:res:res*t])
sc = reshape(scores,(N*N,t))
xt = transpose(x)
sct = transpose(sc)
responseS = zeros(N^2,N^2,n_tau)
for i in ProgressBar(1:n_tau)
    responseS[:,:,i] = cov(xt[i:end,:],sct[1:end-i+1,:])
end

responseS_norm = zeros(N^2,N^2,n_tau)
for i in 1:n_tau
    responseS_norm[:,:,i] = responseS[:,:,i]*inv(responseS[:,:,1])
end

responseS_mean = mean_response(responseS_norm)

pl1 = plot(responseN[1,:].+err[1,:]./sqrt(n_ens),label="numerics",color=:red,title="1 -> 1",xlabel="time",ylabel="response", linewidth=3)
pl1 = plot!(responseN[1,:].-err[1,:]./sqrt(n_ens),label="",color=:red, linewidth=3)
pl1 = plot!(responseL_mean[1,:],label="linear app",color=:blue, linewidth=3)
pl1 = plot!(1:res:n_tau, responseS_mean[1,1:Int(floor(n_tau/res))],label="score",color=:black, linewidth=3)

pl2 = plot(responseN[2,:].+err[2,:]./sqrt(n_ens),label="numerics",color=:red,title="1 -> 2",xlabel="time",ylabel="response", linewidth=3)
pl2 = plot!(responseN[2,:].-err[2,:]./sqrt(n_ens),label="",color=:red, linewidth=3)
pl2 = plot!(responseL_mean[2,:],label="linear app",color=:blue, linewidth=3)
pl2 = plot!(1:res:n_tau, responseS_mean[2,1:Int(floor(n_tau/res))],label="score",color=:black, linewidth=3)

pl = plot(pl1,pl2,layout=(1,2),size=(1000,500))
display(pl)
savefig(pl, "response_gamma_$γ.png")
