module GenerateTrajectories

using Statistics
using Random
using LinearAlgebra
using ProgressBars

export gradient_periodic, laplacian_periodic, corr, trajectory, regularization, s

function gradient_periodic(N,diff_ord; dx=1)
    M1 = [-1,0,1] ./ (2*dx)
    D = zeros(N,N)
    for i = 1:N
        for j in -diff_ord:diff_ord
            if j>=0
                D[i,(i+j-1)%N+1] = M1[abs(j)+2]
            else
                D[i,(i+j-1+N)%N+1] = M1[abs(j+1)+1]
            end
        end
    end 
    return D
end

function laplacian_periodic(N,diff_ord; dx=1)
    M1 = [-2, 1] ./ dx^2
    M2 = [-(5/2), 4/3, -(1/12)] ./ dx^2
    M3 = [-(49/18), 3/2, -(3/20), 1/90] ./ dx^2
    M4 = [-(205/72), 8/5, -(1/5), 8/315, -(1/560)] ./ dx^2
    M5 = [-(5269/1800), 5/3, -(5/21), 5/126, -(5/1008), 1/3150] ./ dx^2
    M6 = [-(5369/1800), 12/7, -(15/56), 10/189, -(1/112), 2/1925, -(1/16632)] ./ dx^2
    M7 = [-(266681/88200), 7/4, -(7/24), 7/108, -(7/528), 7/3300, -(7/30888), 1/84084] ./ dx^2
    M8 = [-(1077749/352800), 16/9, -(14/45), 112/1485, -(7/396), 112/32175,
    -(2/3861), 16/315315, -(1/411840)] ./ dx^2
    M9 = [-(9778141/3175200), 9/5, -(18/55), 14/165, -(63/2860), 18/3575,
    -(2/2145), 9/70070, -(9/777920), 1/1969110] ./ dx^2
    M10 = [-(1968329/635040), 20/11, -(15/44), 40/429, -(15/572), 24/3575,
    -(5/3432), 30/119119, -(5/155584), 10/3741309, -(1/9237800)] ./ dx^2
    M11 = [-(239437889/76839840), 11/6, -(55/156), 55/546, -(11/364), 11/1300,
    -(11/5304), 55/129948, -(55/806208), 11/1360476, -(11/17635800),
    1/42678636] ./ dx^2
    M12 = [-(240505109/76839840), 24/13, -(33/91), 88/819, -(99/2912),
    396/38675, -(11/3978), 132/205751, -(33/268736), 44/2380833,
    -(3/1469650), 12/81800719, -(1/194699232)] ./ dx^2
    M13 = [-(40799043101/12985932960), 13/7, -(13/35), 143/1260, -(143/3808),
    143/11900, -(143/40698), 143/158270, -(143/723520), 13/366282,
    -(13/2600150), 13/25169452, -(13/374421600), 1/878850700] ./ dx^2
    M14 = [-(40931552621/12985932960), 28/15, -(91/240), 91/765, -(1001/24480),
    1001/72675, -(1001/232560), 286/237405, -(91/310080), 182/3008745,
    -(91/8914800), 91/67418175, -(7/53488800), 7/847463175, -(1/3931426800
     )] ./ dx^2
    M15 = [-(205234915681/64929664800), 15/8, -(105/272), 455/3672,
    -(455/10336), 1001/64600, -(715/139536), 195/126616, -(195/475456),
    455/4813992, -(273/14858000), 21/7191272, -(7/19255968), 1/30132024,
    -(1/506717232), 1/17450721000] ./ dx^2

    if diff_ord == 1
        M = M1
    elseif diff_ord == 2
        M = M2
    elseif diff_ord == 3
        M = M3
    elseif diff_ord == 4
        M = M4
    elseif diff_ord == 5
        M = M5
    elseif diff_ord == 6
        M = M6
    elseif diff_ord == 7
        M = M7
    elseif diff_ord == 8
        M = M8
    elseif diff_ord == 9
        M = M9
    elseif diff_ord == 10
        M = M10
    elseif diff_ord == 11
        M = M11
    elseif diff_ord == 12
        M = M12
    elseif diff_ord == 13
        M = M13
    elseif diff_ord == 14
        M = M14
    elseif diff_ord == 15
        M = M15
    end

    DD = zeros(N,N)
    for i = 1:N
        for j in -diff_ord:diff_ord
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
    # Some possiblities for the drift term
    id = I + zeros(N, N)
    # DX = 1
    # X = ([-N/2:-1...,1:N/2...].+1/2)*DX
    # U2x = (ones(N)*transpose(X))
    # U2 = kron(id, U2x) + kron(U2x, id)

    # X4 = 2*(1-alpha).*X + 4*alpha*X.^3
    # U4x = (ones(N)*transpose(X4))
    # U4y = id
    # U4 = kron(id, U4x) + kron(U4y, id) 
    # U = U4

    # gradx = gradient_periodic(N,1)
    # grad = kron(id, gradx) + kron(gradx, id)

    lapx = laplacian_periodic(N,1)
    lap = kron(id, lapx) + kron(lapx, id)
    return ((lap.-beta)* tanh.(gamma*(x))) .- alpha
    
    #(lap.-beta)* x.^3 .- gamma#- gamma*alpha*exp.(alpha*gamma*x)
    
    #1/alpha*((1-gamma)*(lap.-beta)*(alpha.*(x.-0.5)).-gamma*(alpha.*(x.-0.5)).^3)
end

function trajectory(X0,N,t,seed,sigma,Dt,alpha,beta,gamma)
    Random.seed!(seed)
    f(x) = s(x,N,alpha,beta,gamma)
    x = Vector{Float64}[]
    xOld = X0
    xNew = X0
    r = randn(N^2,t)
    ΓL = LinearAlgebra.cholesky(corr(N)).L
    for i in 1:t
        k1 = f(xOld)
        y = xOld + Dt * k1 * 0.5
        k2 = f(y)
        y = xOld + Dt * k2 * 0.5
        k3 = f(y)
        y = xOld + Dt * k3
        k4 = f(y)
        r_corr = copy(r[:,i])
        mul!(r_corr,ΓL,r[:,i])
        xNew += Dt / 6. * (k1 + 2 * k2 + 2 * k3 + k4) + sqrt(Dt) .* (sigma .* r_corr)
        push!(x, xNew)
        xOld = xNew
    end
    X = zeros(Float64,N^2,t)
    X[:,1] = X0[:]
    for i in 2:t
        X[:,i] = x[i][:]
    end
    return X
end

function trajectory2(X0,N,t,seed,sigma,Dt,alpha,beta,gamma,res)
    Random.seed!(seed)
    f(x) = s(x,N,alpha,beta,gamma)
    x = Vector{Float64}[]
    xOld = X0
    xNew = X0
    ΓL = LinearAlgebra.cholesky(corr(N)).L
    for i in 1:t
        r = randn(N^2)
        k1 = f(xOld)
        y = xOld + Dt * k1 * 0.5
        k2 = f(y)
        y = xOld + Dt * k2 * 0.5
        k3 = f(y)
        y = xOld + Dt * k3
        k4 = f(y)
        r_corr = copy(r)
        mul!(r_corr,ΓL,r)
        xNew += Dt / 6. * (k1 + 2 * k2 + 2 * k3 + k4) + sqrt(Dt) .* (sigma .* r_corr)
        if i%res == 0
            push!(x, xNew)
        end
        xOld = xNew
    end
    t_red = length(x)
    X = zeros(Float64,N^2,t_red)
    X[:,1] = X0[:]
    for i in 2:t_red
        X[:,i] = x[i][:]
    end
    return X
end

function regularization(X)
    indices = findall(X.>=1)
    X[indices] .= 1.
    indices = findall(X.<=-1)
    X[indices] .= -1.
    return X
end

function regularization2(X,n_std)
    X = X .- mean(X)
    X = X ./ std(X) ./ n_std
    indices = findall(X.>=1)
    X[indices] .= 1.
    indices = findall(X.<=-1)
    X[indices] .= -1.
    return X
end
end