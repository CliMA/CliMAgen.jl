for inchannel in inchannels
    clims = clims_tuple[inchannel]
    fieldname = names[inchannel]

    # training
    x_lr = train_data_lores[:,:,inchannel,:]
    x_hr = train_data[:,:,inchannel,:]
    s_hr = train_samples[:,:,inchannel,:]
    r_hr = train_random_samples[:,:,inchannel,:]

    if inchannel == precip_channel
        precip_x = x_hr[:,:,inchannel,:][:]
        precip_s =  s_hr[:,:,inchannel,:][:]
        precip_x = precip_x[precip_x .> clims[1]]
        precip_s = precip_s[precip_s .> clims[1]]
        precip_r =  r_hr[:,:,inchannel,:][:]
        precip_r = precip_r[precip_r .> clims[1]]
        pixel_plots(precip_x, precip_s, ["training", "downscaled"], joinpath(savedir,"downscaled_samples_train_$(inchannel).png"), clims, fieldname)
        pixel_plots(precip_x, precip_r, ["training", "random gen"], joinpath(savedir,"random_samples_train_$(inchannel).png"), clims,fieldname)
        simple_histogram_plot(precip_x, precip_s, ["training", "downscaled"], joinpath(savedir,"downscaled_samples_train_$(inchannel)_hist.png"), clims,fieldname,)
        simple_histogram_plot(precip_x, precip_r, ["training", "random gen"], joinpath(savedir,"random_samples_train_$(inchannel)_hist.png"), clims,fieldname,)
    else
        # pixel_plots(x_hr, s_hr, ["training", "downscaled"], joinpath(savedir,"downscaled_samples_train_$(inchannel).png"), clims, fieldname)
        # pixel_plots(x_hr, r_hr, ["training", "random gen"], joinpath(savedir,"random_samples_train_$(inchannel).png"), clims,fieldname)
        simple_histogram_plot(x_hr, s_hr, ["training", "downscaled"], joinpath(savedir,"downscaled_samples_train_$(inchannel)_hist.png"), clims,fieldname,)
        simple_histogram_plot(x_hr, r_hr, ["training", "random gen"], joinpath(savedir,"random_samples_train_$(inchannel)_hist.png"), clims,fieldname,)
    end

    fig = Figure(resolution=(3250, 1200), fontsize=24)
    ax = CairoMakie.Axis(fig[1,1], ylabel="Lo res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, x_lr[:,:,1], clims = clims)
    ax = CairoMakie.Axis(fig[2,1], ylabel="Hi res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, x_hr[:,:,1], clims = clims)
    ax = CairoMakie.Axis(fig[3,1], ylabel="Hi res fake", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, s_hr[:,:,1], clims = clims)
    for i in 2:8
        ax = CairoMakie.Axis(fig[1,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, x_lr[:,:,i*8], clims = clims)
        ax = CairoMakie.Axis(fig[2,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, x_hr[:,:,i*8], clims = clims)
        ax = CairoMakie.Axis(fig[3,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, s_hr[:,:,i*8], clims = clims)
    end
    save(joinpath(savedir,"downscaling_train_$(inchannel).png"), fig, px_per_unit = 2)

    # testing
    x_lr = test_data_lores[:,:,inchannel,:]
    x_hr = test_data[:,:,inchannel,:]
    s_hr = test_samples[:,:,inchannel,:]
    r_hr = test_random_samples[:,:,inchannel,:]
    if inchannel == precip_channel
        precip_x = x_hr[:,:,inchannel,:][:]
        precip_s =  s_hr[:,:,inchannel,:][:]
        precip_x = precip_x[precip_x .> clims[1]]
        precip_s = precip_s[precip_s .> clims[1]]
        precip_r =  r_hr[:,:,inchannel,:][:]
        precip_r = precip_r[precip_r .> clims[1]]
        pixel_plots(precip_x, precip_s, ["test data", "downscaled"], joinpath(savedir,"downscaled_samples_test_$(inchannel).png"), clims,fieldname,)
        pixel_plots(precip_x, precip_r, ["test data", "random gen"], joinpath(savedir,"random_samples_test_$(inchannel).png"), clims,fieldname)
        simple_histogram_plot(precip_x, precip_s, ["test data", "downscaled"], joinpath(savedir,"downscaled_samples_test_$(inchannel)_hist.png"), clims,fieldname,)
        simple_histogram_plot(precip_x, precip_r, ["test data", "random gen"], joinpath(savedir,"random_samples_test_$(inchannel)_hist.png"), clims,fieldname,)
    else
        # pixel_plots(x_hr, s_hr, ["test data", "downscaled"], joinpath(savedir,"downscaled_samples_test_$(inchannel).png"), clims,fieldname,)
        # pixel_plots(x_hr, r_hr, ["test data", "random gen"], joinpath(savedir,"random_samples_test_$(inchannel).png"), clims,fieldname)
        simple_histogram_plot(x_hr, s_hr, ["test data", "downscaled"], joinpath(savedir,"downscaled_samples_test_$(inchannel)_hist.png"), clims,fieldname,)
        simple_histogram_plot(x_hr, r_hr, ["test data", "random gen"], joinpath(savedir,"random_samples_test_$(inchannel)_hist.png"), clims,fieldname,)

    end

    fig = Figure(resolution=(3250, 1200), fontsize=24)
    ax = CairoMakie.Axis(fig[1,1], ylabel="Lo res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, x_lr[:,:,1], clims = clims)
    ax = CairoMakie.Axis(fig[2,1], ylabel="Hi res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, x_hr[:,:,1], clims = clims)
    ax = CairoMakie.Axis(fig[3,1], ylabel="Hi res fake", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, s_hr[:,:,1], clims = clims)
    for i in 2:8
        ax = CairoMakie.Axis(fig[1,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, x_lr[:,:,i*8], clims = clims)
        ax = CairoMakie.Axis(fig[2,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, x_hr[:,:,i*8], clims = clims)
        ax = CairoMakie.Axis(fig[3,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, s_hr[:,:,i*8], clims = clims)
    end
    save(joinpath(savedir,"downscaling_test_$(inchannel).png"), fig, px_per_unit = 2)
end


