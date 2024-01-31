using GLMakie
using HDF5

# run from giorgini2d
# all response data is in the data folder
#=
data_directory = "data/"

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

    lr = linear_response[1:8:end,1, :] # effect of perturbation of pixel 1 on pixel i
    srh = score_response_hack[1:8:end,1, :] # effect of perturbation of pixel 1 on pixel i
    nr = numerical_response[1:8:end, 1:4:end] # effect of perturbation of pixel 1 on pixel i
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

=#

data_directory = "/home/sandre/Repositories/CliMAgen.jl/examples/giorgini2d/" * "data/"

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
# for (i, gl) in enumerate(glist)
i = 2
gl = gl_middle
    # load files
    hfile = h5open(filestrings[i,1], "r")
    linear_response = read(hfile["response"])
    close(hfile)
    hfile = h5open(filestrings[i, 2], "r")
    score_response = read(hfile["response"])
    score_response_hack = read(hfile["both normalized response"])
    close(hfile)
    # hfile = h5open(filestrings[i, 3], "r")
    # numerical_response = read(hfile["pixel response"])
    # close(hfile)

    lr = [linear_response[end-64*i + 1, 1, j]  for i in 1:64, j in 1:41]# linear_response[1:64:end,1, :] # effect of perturbation of pixel 1 on pixel i
    srh = [score_response_hack[end-64*i + 1, 1, j]  for i in 1:64, j in 1:41] # score_response_hack[1:64:end, 1, :] # effect of perturbation of pixel 1 on pixel i
    # nr = numerical_response[1:64:end, 1:4:end] # effect of perturbation of pixel 1 on pixel i
    for j in 1:8
        ax = Axis(gl[1, j])
        if j == 1
            jj = j
        else
            jj  = 10 - j
        end
        if (i == 1) && (j == 1)
            # lines!(ax, nr[jj,:], color=:black, label = "Truth")
            lines!(ax, lr[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh[jj,:], color=(:orange, 0.5), linewidth = lw)
            axislegend(ax; position=:rt, labelsize= scale * 16)
        elseif (i == 1) && (j == 2)
            # lines!(ax, nr[jj,:], color=:black)
            lines!(ax, lr[jj,:], color=(:blue, 0.5), linewidth = lw, label = "Linear")
            lines!(ax, srh[jj,:], color=(:orange, 0.5), linewidth = lw)
            axislegend(ax; position=:rt, labelsize= scale * 16)
        elseif (i == 1) && (j == 3)
            # lines!(ax, nr[jj,:], color=:black)
            lines!(ax, lr[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh[jj,:], color=(:orange, 0.5), linewidth = lw, label = "Score")
            axislegend(ax; position=:rt, labelsize= scale * 16)
        else
            # lines!(ax, nr[jj,:], color=:black)
            lines!(ax, lr[jj,:], color=(:blue, 0.5), linewidth = lw)
            lines!(ax, srh[jj,:], color=(:orange, 0.5), linewidth = lw)
        end
    end
    Label(gl[1, 0], text = titles[i], fontsize = scale * 50, rotation = pi/2)
# end
display(fig)

##
fig = Figure()
for i in 1:16
    ii = (i-1) % 4 + 1
    jj = (i-1) ÷ 4 + 1
    ax = Axis(fig[ii, jj]; title = "t = $(i-1)")
    heatmap!(ax, reshape(score_response_hack[:, 32 + 64 * 32, i], (64, 64)), colorrange = (0, 0.1), colormap = :afmhot)
end