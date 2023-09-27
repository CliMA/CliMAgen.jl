module Tools

using Statistics
using Random
using LinearAlgebra
using StatsBase

export laplacian_periodic, corr, s, regularization

function laplacian_periodic(N; dx=1)
    M = [-2, 1] ./ dx^2
    DD = zeros(N,N)
    for i = 1:N
        for j in -1:1
            if j>=0
                DD[(i+j-1)%N+1,i] = M[abs(j)+1]
            else
                DD[(i+j-1+N)%N+1,i] = M[abs(j)+1]
            end
        end
    end 
    return DD 
end

function corr(N)
    corr = reshape(zeros(N^4), (N^2,N^2))
    for i1 in 1:N
        for j1 in 1:N
            k1 = (j1-1)*N+i1
            for i2 in 1:N
                for j2 in 1:N
                    k2 = (j2-1)*N+i2
                    corr[k1,k2] = 1.0/sqrt(min(abs(i1-i2),N-abs(i1-i2))^2 + min(abs(j1-j2),N-abs(j1-j2))^2+1)
                end
            end
        end
    end
    return corr
end

function s(x,N,alpha, beta, gamma)
    id = I + zeros(N, N)
    lapx = laplacian_periodic(N)
    lap = kron(id, lapx) + kron(lapx, id)
    return ((lap.-beta)* tanh.(gamma*(x))) .- alpha
end

function regularization(X,n_std)
    X = X .- mean(X)
    X = X ./ std(X) ./ n_std
    indices = findall(X.>=1)
    X[indices] .= 1.
    indices = findall(X.<=-1)
    X[indices] .= -1.
    return X
end
end