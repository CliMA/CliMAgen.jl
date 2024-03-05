using CairoMakie
using HDF5
include("utils.jl")
include("pixel_plots.jl")
include("../../utils_analysis.jl")

# Unique to this experiment
output_dir = "output_tmax_zmuv_lowpass"
fieldnames = ["Temperature", "Precipitation"]
clims_tuple = ((250,330),(-9,-2))
inchannels = [1]
precip_channel = 2
log10precip_floor = -9

savedir = "../$(output_dir)/downscaling"
hdf5_path = joinpath(savedir,"coarse_res_downscale_era_t0p6.hdf5")
fid = HDF5.h5open(hdf5_path, "r")
samples = HDF5.read(fid["downscaled_samples"]);
data = HDF5.read(fid["data_lores"]);
data_hires = HDF5.read(fid["data_hires"])
close(fid)

n_boot = 1000
cil = 0.9
x = (1:64)
min_x = 2^0
max_x = 2^6
min_y = 1e-6
max_y = 4
compute_spectra = x -> hcat(power_spectrum2d(x)...)

for inchannel in inchannels
    clims = clims_tuple[inchannel]
    fieldname = fieldnames[inchannel]
    if inchannel == precip_channel
        precip_data = data[:,:,inchannel,:][:]
        precip_samples =  samples[:,:,inchannel,:][:]
        precip_data = precip_data[precip_data .> log10precip_floor]
        precip_samples = precip_samples[precip_samples .> log10precip_floor]
        pixel_plots(precip_data, precip_samples, ["Low res", "Downscaled"], joinpath(savedir,"coarse_res_downscaled_$(inchannel).png"), clims,fieldname)
        simple_histogram_plot(precip_data, precip_samples, ["Low res", "Downscaled"], joinpath(savedir,"coarse_res_downscaled_$(inchannel)_hist.png"), clims,fieldname,)
    else
        #pixel_plots(data[:,:,inchannel,:], samples[:,:,inchannel,:], data_hires[:,:,inchannel,:], ["Low res", "Downscaled", "Hi res"], joinpath(savedir,"coarse_res_downscaled_$(inchannel).png"), clims,fieldname)
        simple_histogram_plot(samples[:,:,inchannel,:], data_hires[:,:,inchannel,:], ["Downscaled", "Hi res"], joinpath(savedir,"coarse_res_downscaled_$(inchannel)_hist.png"), clims,fieldname,)
    end


    fig = Figure(resolution=(3250, 1200), fontsize=24)
    ax = CairoMakie.Axis(fig[1,1], ylabel="Lo res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, data[:,:,inchannel,1], clims = clims)
    ax = CairoMakie.Axis(fig[2,1], ylabel="Downscaled", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, samples[:,:,inchannel,1], clims = clims)
    ax = CairoMakie.Axis(fig[3,1], ylabel="Hi res data", yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
    CairoMakie.heatmap!(ax, data_hires[:,:,inchannel,1], clims = clims)
    for i in 2:8
        ax = CairoMakie.Axis(fig[1,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, data[:,:,inchannel,i*8], clims = clims)
        ax = CairoMakie.Axis(fig[2,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, samples[:,:,inchannel,i*8], clims = clims)
        ax = CairoMakie.Axis(fig[3,i], yticksvisible=false, xticksvisible=false, xticklabelsvisible=false, yticklabelsvisible=false)
        CairoMakie.heatmap!(ax, data_hires[:,:,inchannel,i*8], clims = clims)
    end

    save(joinpath(savedir,"coarse_res_downscaled_$(inchannel)_img.png"), fig, px_per_unit = 2)

    spectra_data = mapslices(compute_spectra, data[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    spectra_downscaled = mapslices(compute_spectra, samples[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    spectra_hires = mapslices(compute_spectra, data_hires[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    nsamples = size(spectra_data)[2]
    real_l_lr, real_u_lr = get_spectra_bci([Array(spectra_data[:,k]) for k in 1:nsamples], n_boot, cil)
    gen_l_hr_ds, gen_u_hr_ds = get_spectra_bci([Array(spectra_downscaled[:,k]) for k in 1:nsamples], n_boot, cil)
    real_l_hr, real_u_hr = get_spectra_bci([Array(spectra_hires[:,k]) for k in 1:nsamples], n_boot, cil)
   
    fig = Figure(resolution=(600, 600), fontsize=24)

    ax = CairoMakie.Axis(fig[1,1], xlabel="Wavenumber", ylabel="Average power spectrum", title=fieldname, xscale = log2, yscale = log10)
    band!(x, real_l_lr, real_u_lr, color=(:orange, 0.3), label="real low res.")
    lines!(x, real_l_lr, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, real_u_lr, color=(:orange, 0.5), strokewidth = 1.5)
    band!(x, gen_l_hr_ds, gen_u_hr_ds, color=(:purple, 0.3), label="downscaled")
    lines!(x, gen_l_hr_ds, color=(:purple, 0.5), strokewidth = 1.5)
    lines!(x, gen_u_hr_ds, color=(:purple, 0.5), strokewidth = 1.5)
    band!(x, real_l_hr, real_u_hr, color=(:green, 0.3), label="real high res.")
    lines!(x, real_l_hr, color=(:green, 0.5), strokewidth = 1.5)
    lines!(x, real_u_hr, color=(:green, 0.5), strokewidth = 1.5)
    CairoMakie.xlims!(ax, min_x, max_x)
    CairoMakie.ylims!(ax, min_y, max_y)

    axislegend(; position= :rt, labelsize= 16)

    save(joinpath(savedir,"coarse_res_downscaled_$(inchannel)_spectra.png"), fig, px_per_unit = 2)
end
