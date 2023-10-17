using CairoMakie
using HDF5
using DelimitedFiles
using Bootstrap
using Statistics
using CliMADatasets
using StatsBase
using KernelDensity

function get_pdf(data, min_x, max_x, n_grid)
    estimate = kde(data)
    pdf(estimate, LinRange(min_x, max_x, n_grid))
end
function get_spectra_bci(data, n_boot, cil)
    bs = bootstrap(mean, data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end

function get_pdf_bci(data, min_x, max_x, n_grid, n_boot, cil)
    x = LinRange(min_x, max_x, n_grid)
    bs = bootstrap(x -> get_pdf(x, min_x, max_x, n_grid), data, BasicSampling(n_boot))
    ci = confint(bs, PercentileConfInt(cil))
    lower = [c[2] for c in ci]
    upper = [c[3] for c in ci]
    return lower, upper
end


# spectra plot - epoch 200
function spectra_plot(resolution;ch = 1, epoch = 200.0)
    fig = Figure(resolution=(1600, 375), fontsize=24)
    n_boot = 100
    cil = 0.90
    x = (1:div(resolution,2))
    min_x = 2^0
    max_x = div(resolution,2)
    min_y = 1e-6
    max_y = ch == 1 ? 1e0 : 1e1
    # Clip to n_boot samples
    real_stats = readdlm("../stats/train_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
    real_spectra = [real_stats[k, 5:end] for k in 1:n_boot]
    real_l, real_u = get_spectra_bci(real_spectra, n_boot, cil)

    ema = false
    bypass = false
    fake_stats = readdlm("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
    fake_spectra = [fake_stats[k,5:end] for k in 1:n_boot]
    fake_l, fake_u = get_spectra_bci(fake_spectra, n_boot, cil)
    ax = Axis(fig[1,1], xlabel="Wavenumber", ylabel="Average power spectrum", title="Baseline net, no EMA", xscale = log2, yscale = log10)
    band!(x, real_l, real_u, color=(:orange, 0.3), label="Data")
    lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
    band!(x, fake_l, fake_u, color=(:purple, 0.3), label="Generated")
    lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
    xlims!(ax, min_x, max_x)
    ylims!(ax, min_y, max_y)

    ema = true
    bypass = false
    fake_stats = readdlm("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
    fake_spectra = [fake_stats[k,5:end] for k in 1:n_boot]
    fake_l, fake_u = get_spectra_bci(fake_spectra, n_boot, cil)
    ax = Axis(fig[1,2], xlabel="Wavenumber", title="Baseline net, EMA", xscale = log2, yscale = log10, yticklabelsvisible = false, yticksvisible = false)
    band!(x, real_l, real_u, color=(:orange, 0.3), label="Data")
    lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
    band!(x, fake_l, fake_u, color=(:purple, 0.3), label="Generated")
    lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
    xlims!(ax, min_x, max_x)
    ylims!(ax, min_y, max_y)

    ema = false
    bypass = true
    fake_stats = readdlm("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
    fake_spectra = [fake_stats[k,5:end] for k in 1:n_boot]
    fake_l, fake_u = get_spectra_bci(fake_spectra, n_boot, cil)
    ax = Axis(fig[1,3], xlabel="Wavenumber", title="Modified net, no EMA", xscale = log2, yscale = log10, yticklabelsvisible = false, yticksvisible = false)
    band!(x, real_l, real_u, color=(:orange, 0.3), label="Data")
    lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
    band!(x, fake_l, fake_u, color=(:purple, 0.3), label="Generated")
    lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
    xlims!(ax, min_x, max_x)
    ylims!(ax, min_y, max_y)

    ema = true
    bypass = true
    fake_stats = readdlm("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
    fake_spectra = [fake_stats[k,5:end] for k in 1:n_boot]
    fake_l, fake_u = get_spectra_bci(fake_spectra, n_boot, cil)
    ax = Axis(fig[1,4], xlabel="Wavenumber", title="Modified net, EMA", xscale = log2, yscale = log10, yticklabelsvisible = false, yticksvisible = false)
    band!(x, real_l, real_u, color=(:orange, 0.3), label="Data")
    lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
    band!(x, fake_l, fake_u, color=(:purple, 0.3), label="Generated")
    lines!(x, fake_l, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(x, fake_u, color=(:purple, 0.5), strokewidth = 1.5)
    xlims!(ax, min_x, max_x)
    ylims!(ax, min_y, max_y)

    axislegend(; position= :rt, labelsize= 16)

    save("fig_spectra_ch$(ch)_$resolution.png", fig, px_per_unit = 2)
end

# images

function image_plot(resolution;ch = 1, epoch = 200.0)
    fig = Figure(resolution=(1850, 375), fontsize=24)
    fid = h5open("../stats/train_$resolution/samples.hdf5","r")
    data = read(fid, "samples")
    close(fid)

    clims = extrema(data[:,:,ch,:][:])
    ema = false
    bypass = false
    fid = h5open("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/samples.hdf5","r")
    snapshot = read(fid, "samples")
    close(fid)
    ax = Axis(fig[1,1], ylabel="", title="Baseline net, no EMA", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
    co = heatmap!(ax, snapshot[:,:,ch, 1].-mean(snapshot[:,:,ch, 1]), colormap=Reverse(:blues),clims = clims, colorbar = false)

    ema = true
    bypass = false
    fid = h5open("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/samples.hdf5","r")
    snapshot = read(fid, "samples")
    close(fid)
    ax = Axis(fig[1,2], ylabel="", title="Baseline net, EMA", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
    co = heatmap!(ax, snapshot[:,:,ch,1].-mean(snapshot[:,:,ch, 1]), colormap=Reverse(:blues),clims = clims, colorbar = false)

    ema = false
    bypass = true
    fid = h5open("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/samples.hdf5","r")
    snapshot = read(fid, "samples")
    close(fid)
    ax = Axis(fig[1,3], ylabel="", title="Modified net, no EMA", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
    co = heatmap!(ax, snapshot[:,:,ch,1].-mean(snapshot[:,:,ch, 1]), colormap=Reverse(:blues),clims = clims, colorbar = false)

    ema = true
    bypass = true
    fid = h5open("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/samples.hdf5","r")
    snapshot = read(fid, "samples")
    close(fid)
    ax = Axis(fig[1,4], ylabel="", title="Modified net, EMA", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
    co = heatmap!(ax, snapshot[:,:,ch,1].-mean(snapshot[:,:,ch, 1]), colormap=Reverse(:blues),clims = clims, colorbar = false)
    
    ax = Axis(fig[1,5], ylabel="", title="Data", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false)
    co = heatmap!(ax, data[:,:,ch, 1].-mean(data[:,:,ch, 1]), colormap=Reverse(:blues),clims = clims, colorbar = false)


    save("fig_snapshots_$(ch)_$resolution.png", fig, px_per_unit = 2)
end


function spatial_mean_plot(resolution; ch=1)
    fig = Figure(resolution=(1600, 400), fontsize=24)
    ax1 = Axis(fig[1,1],xlabel="Epochs",  ylabel="Spatial Mean", title="Baseline net, no EMA", xticks = (2:1:5, ["80", "120", "160", "200"]))
    ax2 = Axis(fig[1,2], xlabel="Epochs", title="Baseline net, EMA",xticks = (2:1:5, ["80", "120", "160", "200"]))
    ax3 = Axis(fig[1,3],xlabel="Epochs",  title="Modified net, no EMA",xticks = (2:1:5, ["80", "120", "160", "200"]))
    ax4 = Axis(fig[1,4],xlabel="Epochs",  title="Modified net, EMA",xticks = (2:1:5, ["80", "120", "160", "200"]))

    N_fake = 100
    real_stats = readdlm("../stats/train_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
    real_spatial_means = real_stats[1:N_fake,1]
    label = vcat([:real for k in 1:length(real_spatial_means)], [:fake for k in 1:N_fake])
    side = vcat([:left for k in 1:length(real_spatial_means)], [:right for k in 1:N_fake])

    color = @. ifelse(label === :real, :orange, :teal)

    bypass=false
    ema = false
    for epoch in [80.0,120.0, 160.0, 200.0]
        fake_stats = readdlm("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
        fake_spatial_means = fake_stats[:,1]
        ys = vcat(real_spatial_means, fake_spatial_means)
        xs = zeros(length(ys)) .+ div(epoch, 40)
        violin!(ax1, xs, ys, side = side, color = color)
    end

    bypass=false
    ema = true
    for epoch in [80.0,120.0, 160.0, 200.0]
        fake_stats = readdlm("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
        fake_spatial_means = fake_stats[:,1]
        @info Statistics.std(fake_spatial_means)
        ys = vcat(real_spatial_means, fake_spatial_means)
        xs = zeros(length(ys)) .+ div(epoch, 40)
        violin!(ax2, xs, ys, side = side, color = color)
    end

    bypass=true
    ema = false
    for epoch in [80.0,120.0, 160.0, 200.0]
        fake_stats = readdlm("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
        fake_spatial_means = fake_stats[:,1]
        ys = vcat(real_spatial_means, fake_spatial_means)
        xs = zeros(length(ys)) .+ div(epoch, 40)
        violin!(ax3, xs, ys, side = side, color = color)
    end
    bypass=true
    ema = true
    for epoch in [80.0,120.0, 160.0, 200.0]
        fake_stats = readdlm("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
        fake_spatial_means = fake_stats[:,1]
        ys = vcat(real_spatial_means, fake_spatial_means)
        xs = zeros(length(ys)) .+ div(epoch, 40)
        violin!(ax4, xs, ys, side = side, color = color)
    end
    save("fig_means_$(ch)_$(resolution).png", fig, px_per_unit = 2)

end


function image_plot2(;resolution = 512, ch = 2, epoch = 200.0)

    fid = h5open("../stats/train_$resolution/samples.hdf5","r")
    data_snapshots = read(fid, "samples")
    close(fid)
    clims = extrema(data_snapshots[:,:,ch,:][:])
    ema = false
    bypass = false
    fid = h5open("../stats/ema_$(ema)_bypass_$(bypass)_epoch_$(epoch)/gen_$(resolution)/samples.hdf5","r")
    snapshot = read(fid, "samples")
    close(fid)
    xtrain = CliMADatasets.Turbulence2DContext(:train; fraction = 1.0f0, resolution=512, wavenumber = 1.0f0, Tx=FT,)[:];
    #real_stats = readdlm("../stats/train_$(resolution)/statistics_ch$(ch)_1.0.csv",',')
    #real_spatial_means = real_stats[:,1]
    real_spatial_means = Statistics.mean(xtrain, dims = (1,2))[1,1,ch,:];
    minx = -3e-7
    maxx = 3e-7
    ngrid = 20
    nboot = 100
    cil = 0.9

    μ = mean(snapshot[:,:,ch, 2])

    fig = Figure(resolution=(1500, 500), fontsize=24)
    ax = Axis(fig[1,1], ylabel="", title="Vanilla Model", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, titlefont = :regular)
    co = heatmap!(ax, snapshot[:,:,ch, 2].-μ, colormap=Reverse(:blues),clims = clims, colorbar = false)

    ax = Axis(fig[1,2], ylabel="", title="Data", xticklabelsvisible = false, yticklabelsvisible = false, xticksvisible = false, yticksvisible = false, titlefont = :regular)
    co = heatmap!(ax, data_snapshots[:,:,ch, 1].-mean(data_snapshots[:,:,ch, 1]), colormap=Reverse(:blues),clims = clims, colorbar = false)

    real_l, real_u= get_pdf_bci(real_spatial_means, minx, maxx, ngrid, nboot, cil)
    x = LinRange(minx, maxx, ngrid)
    sm_pdf = get_pdf(real_spatial_means, minx, maxx, ngrid)
    ax = Axis(fig[1,3],xlabel="Spatial Mean (x 1e-7)",  ylabel="Probability Density",xticks = (-3e-7:3e-7:6e-7, ["-3", "0", "3", "6"]))
    band!(x, real_l, real_u, color=(:orange, 0.3))
    lines!(x, real_l, color=(:orange, 0.5), strokewidth = 1.5, label="Data Distribution")
    lines!(x, real_u, color=(:orange, 0.5), strokewidth = 1.5)
    #lines!(x, sm_pdf, color=(:orange, 0.5), label="Data Distribution",strokewidth = 1.5)
    co = scatter!([μ, μ], [0,0], color=(:purple, 0.5), label = "Generated Sample")
    save("fig_motivating_example_$(ch).png", fig, px_per_unit = 2)
end


resolution = 512
spatial_mean_plot(resolution; ch = 1)
spectra_plot(resolution;ch = 1, epoch = 200.0)
image_plot(resolution;ch = 1, epoch = 200.0)
spectra_plot(resolution;ch = 2, epoch = 200.0)
image_plot(resolution;ch = 2, epoch = 200.0)
image_plot2(;resolution = 512, ch = 2, epoch = 200.0)