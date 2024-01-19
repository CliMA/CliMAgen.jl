using GLMakie
using HDF5
using Distributions
using LinearAlgebra

function generate_2d_gaussian_lattice(N,mu,sigma)
    center = (N+1) / 2
    lattice = [exp(-0.5 * ([i, j] .- center)' * inv(sigma) * ([i, j] .- center)) for i in 1:N, j in 1:N]
   
    norm_factor = sqrt(sum(lattice .^ 2))
    normalized_lattice = lattice / norm_factor

    shift_x = Int(round(mod(mu[1] - center+0.5, N)))
    shift_y = Int(round(mod(mu[2] - center+0.5, N)))
    shifted_lattice = circshift(normalized_lattice, (shift_x, shift_y))

    return shifted_lattice
end

function create_response_smooth(dx0,lags,R)
    N = length(dx0[:,1])
    R_smooth = zeros(N^2, length(lags))
    dx0_1D = reshape(dx0, N^2)
    # dx0_1D = ones(N^2)/N^2
    for t in 1:length(lags)
        for i in 1:N^2
            R_smooth[i, t] = vcat(R[i:end, t],R[1:i-1, t])'*dx0_1D
        end
    end
    return R_smooth
end

# run from giorgini2d
# all response data is in the data folder

data_directory = "responses/"

response_strings = reshape(["linear_response_", "score_response_", "numerical_response_"], 1, :)
label_strings = reshape(["0.3_0.5_0.1_2.0.hdf5", "0.3_0.5_1.0_2.0.hdf5", "0.3_0.5_10.0_2.0.hdf5"], :, 1)

filestrings = data_directory .* response_strings .* label_strings
# filestrings[3,:]


scale = 0.75
fig = Figure(resolution = (scale * 2000, scale * 750)) 
gl_top = GridLayout(fig[1,1]; title = "Weak")
gl_middle = GridLayout(fig[2,1]; title = "Medium")
gl_bottom = GridLayout(fig[3,1]; title = "Strong")
titles = reverse(["Strong", "Medium", "Weak"])
lw = 4 # linewidth
glist = reverse([gl_bottom, gl_middle, gl_top])

for (i, gl) in enumerate(glist)
    # load files
    hfile = h5open(filestrings[i,1], "r")
    linear_response = read(hfile["response"])
    close(hfile)
    hfile = h5open(filestrings[i, 2], "r")
    score_response = read(hfile["response"])
    score_response_hack = read(hfile["both normalized response"])
    close(hfile)
    hfile = h5open(filestrings[i, 3], "r")
    numerical_response = read(hfile["pixel response"])
    close(hfile)

    lr = linear_response[5:8:end,1, :] # effect of perturbation of pixel 1 on pixel i
    srh = score_response_hack[5:8:end,1, :] # effect of perturbation of pixel 1 on pixel i
    nr = numerical_response[5:8:end, 1:4:end] # effect of perturbation of pixel 1 on pixel i
    for j in 1:8
        ax = Axis(gl[1, j])
        if j == 1
            jj = j
        else
            jj  = 10 - j
        end
        if (i == 1) && (j == 1)
            lines!(ax, nr[jj,:], color=:black, label = "Truth")
            lines!(ax, lr[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh[jj,:], color=(:orange, 0.5), linewidth = lw)
            axislegend(ax; position=:rt, labelsize= scale * 16)
        elseif (i == 1) && (j == 2)
            lines!(ax, nr[jj,:], color=:black)
            lines!(ax, lr[jj,:], color=(:blue, 0.5), linewidth = lw, label = "Linear")
            lines!(ax, srh[jj,:], color=(:orange, 0.5), linewidth = lw)
            axislegend(ax; position=:rt, labelsize= scale * 16)
        elseif (i == 1) && (j == 3)
            lines!(ax, nr[jj,:], color=:black)
            lines!(ax, lr[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh[jj,:], color=(:orange, 0.5), linewidth = lw, label = "Score")
            axislegend(ax; position=:rt, labelsize= scale * 16)
        else
            lines!(ax, nr[jj,:], color=:black)
            lines!(ax, lr[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh[jj,:], color=(:orange, 0.5), linewidth = lw)
        end
    end
    Label(gl[1, 0], text = titles[i], fontsize = scale * 50, rotation = pi/2)
end
display(fig)

##
N = 8
mu = [1 1]
sigma = [1 0; 0 1]
dx0 = generate_2d_gaussian_lattice(N,mu,sigma)
x = range(1, stop=N, length=N)
y = range(1, stop=N, length=N)

fig = Figure()
ax = Axis3(fig[1, 1], aspect = (1, 1, 1))
surface!(ax, x, y, dx0, colormap = :viridis, rotation = pi/2)
save("perturbation_$(sigma[1,1]).png", fig)

scale = 0.75
fig = Figure(resolution = (scale * 2000, scale * 750)) 
gl_top = GridLayout(fig[1,1]; title = "Weak")
gl_middle = GridLayout(fig[2,1]; title = "Medium")
gl_bottom = GridLayout(fig[3,1]; title = "Strong")
titles = reverse(["Strong", "Medium", "Weak"])
lw = 4 # linewidth
glist = reverse([gl_bottom, gl_middle, gl_top])

for (i, gl) in enumerate(glist)
    # load files
    hfile = h5open(filestrings[i,1], "r")
    linear_response = read(hfile["response"])
    linear_indices = read(hfile["lag_indices"])
    close(hfile)
    hfile = h5open(filestrings[i, 2], "r")
    score_response = read(hfile["response"])
    score_response_hack = read(hfile["both normalized response"])
    score_indices = read(hfile["lag_indices"])
    close(hfile)
    hfile = h5open(filestrings[i, 3], "r")
    numerical_response = read(hfile["pixel response"])
    numerical_indices = read(hfile["lag_indices"])
    close(hfile)

    lr_smooth = create_response_smooth(dx0,linear_indices,linear_response[:,1,:])[1:8:end,1:27] # effect of perturbation of pixel 1 on pixel i
    srh_smooth = create_response_smooth(dx0,score_indices,score_response_hack[:,1,:])[1:8:end,1:27] # effect of perturbation of pixel 1 on pixel i
    nr_smooth = create_response_smooth(dx0,numerical_indices,numerical_response[:,:])[1:8:end,1:4:4*27] # effect of perturbation of pixel 1 on pixel i
    
    for j in 1:8
        ax = Axis(gl[1, j])
        if j == 1
            jj = j
        else
            jj  = 10 - j
        end
        if (i == 1) && (j == 1)
            lines!(ax, nr_smooth[jj,:], color=:black, label = "Truth")
            lines!(ax, lr_smooth[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh_smooth[jj,:], color=(:orange, 0.5), linewidth = lw)
            axislegend(ax; position=:rt, labelsize= scale * 16)
        elseif (i == 1) && (j == 2)
            lines!(ax, nr_smooth[jj,:], color=:black)
            lines!(ax, lr_smooth[jj,:], color=(:blue, 0.5), linewidth = lw, label = "Linear")
            lines!(ax, srh_smooth[jj,:], color=(:orange, 0.5), linewidth = lw)
            axislegend(ax; position=:rt, labelsize= scale * 16)
        elseif (i == 1) && (j == 3)
            lines!(ax, nr_smooth[jj,:], color=:black)
            lines!(ax, lr_smooth[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh_smooth[jj,:], color=(:orange, 0.5), linewidth = lw, label = "Score")
            axislegend(ax; position=:rt, labelsize= scale * 16)
        else
            lines!(ax, nr_smooth[jj,:], color=:black)
            lines!(ax, lr_smooth[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh_smooth[jj,:], color=(:orange, 0.5), linewidth = lw)
        end
    end
    Label(gl[1, 0], text = titles[i], fontsize = scale * 50, rotation = pi/2)
end
save("responses_smooth_$(sigma[1,1]).png", fig)

