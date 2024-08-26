using HDF5
using CairoMakie
include("../../utils_analysis.jl")

n_boot = 1000
cil = 0.9
x = (1:64)
min_x = 2^0
max_x = 2^6
min_y = 1e-6
max_y = 4
compute_spectra = x -> hcat(power_spectrum2d(x)...)
for inchannel in inchannels
    nsamples = size(train_data_lores)[end]
    x_lr = train_data_lores
    x_hr = train_data
    s_hr = train_samples
    r_hr = train_random_samples

    spectra_lr = mapslices(compute_spectra, x_lr[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    spectra_hr_real = mapslices(compute_spectra, x_hr[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    spectra_hr_downscaled = mapslices(compute_spectra, s_hr[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    spectra_hr_random_gen = mapslices(compute_spectra, r_hr[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]

    real_l_lr, real_u_lr = get_spectra_bci([Array(spectra_lr[:,k]) for k in 1:nsamples], n_boot, cil)
    real_l_hr, real_u_hr = get_spectra_bci([Array(spectra_hr_real[:,k]) for k in 1:nsamples], n_boot, cil)
    gen_l_hr_ds, gen_u_hr_ds = get_spectra_bci([Array(spectra_hr_downscaled[:,k]) for k in 1:nsamples], n_boot, cil)
    gen_l_hr, gen_u_hr = get_spectra_bci([Array(spectra_hr_random_gen[:,k]) for k in 1:nsamples], n_boot, cil)


    fig = Figure(resolution=(600, 600), fontsize=24)

    ax = CairoMakie.Axis(fig[1,1], xlabel="Wavenumber", ylabel="Average power spectrum", title=names[inchannel], xscale = log2, yscale = log10)
    band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
    lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
    # band!(x, gen_l_hr_ds, gen_u_hr_ds, color=(:purple, 0.3), label="downscaled high res.")
    # lines!(x, gen_l_hr_ds, color=(:purple, 0.5), strokewidth = 1.5)
    # lines!(x, gen_u_hr_ds, color=(:purple, 0.5), strokewidth = 1.5)
    band!(x, gen_l_hr, gen_u_hr, color=(:blue, 0.3), label="random gen high res.")
    lines!(x, gen_l_hr, color=(:blue, 0.5), strokewidth = 1.5)
    lines!(x, gen_u_hr, color=(:blue, 0.5), strokewidth = 1.5)
    # band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
    # lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
    # lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
    CairoMakie.xlims!(ax, min_x, max_x)
    CairoMakie.ylims!(ax, min_y, max_y)

    axislegend(; position= :rt, labelsize= 16)

    save(joinpath(savedir,"spectra_train_$(inchannel).png"), fig, px_per_unit = 2)


    # Test


    x_lr = test_data_lores
    x_hr = test_data
    s_hr = test_samples
    r_hr = test_random_samples

    spectra_lr = mapslices(compute_spectra, x_lr[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    spectra_hr_real = mapslices(compute_spectra, x_hr[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    spectra_hr_downscaled = mapslices(compute_spectra, s_hr[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]
    spectra_hr_random_gen = mapslices(compute_spectra, r_hr[:,:,[inchannel],:], dims=[1, 2])[:, 1, 1, :]

    real_l_lr, real_u_lr = get_spectra_bci([Array(spectra_lr[:,k]) for k in 1:nsamples], n_boot, cil)
    real_l_hr, real_u_hr = get_spectra_bci([Array(spectra_hr_real[:,k]) for k in 1:nsamples], n_boot, cil)
    gen_l_hr_ds, gen_u_hr_ds = get_spectra_bci([Array(spectra_hr_downscaled[:,k]) for k in 1:nsamples], n_boot, cil)
    gen_l_hr, gen_u_hr = get_spectra_bci([Array(spectra_hr_random_gen[:,k]) for k in 1:nsamples], n_boot, cil)


    fig = Figure(resolution=(600, 600), fontsize=24)

    ax = CairoMakie.Axis(fig[1,1], xlabel="Wavenumber", ylabel="Average power spectrum", title=names[inchannel], xscale = log2, yscale = log10)
    band!(x, real_l_hr, real_u_hr, color=(:orange, 0.3), label="real high res.")
    lines!(x, real_l_hr, color=(:orange, 0.5), strokewidth = 1.5)
    lines!(x, real_u_hr, color=(:orange, 0.5), strokewidth = 1.5)
    # band!(x, gen_l_hr_ds, gen_u_hr_ds, color=(:purple, 0.3), label="downscaled high res.")
    # lines!(x, gen_l_hr_ds, color=(:purple, 0.5), strokewidth = 1.5)
    # lines!(x, gen_u_hr_ds, color=(:purple, 0.5), strokewidth = 1.5)
    band!(x, gen_l_hr, gen_u_hr, color=(:blue, 0.3), label="random gen high res.")
    lines!(x, gen_l_hr, color=(:blue, 0.5), strokewidth = 1.5)
    lines!(x, gen_u_hr, color=(:blue, 0.5), strokewidth = 1.5)
    # band!(x, real_l_lr, real_u_lr, color=(:green, 0.1), label="real low res.")
    # lines!(x, real_l_lr, color=(:green, 0.2), strokewidth = 1.5)
    # lines!(x, real_u_lr, color=(:green, 0.2), strokewidth = 1.5)
    CairoMakie.xlims!(ax, min_x, max_x)
    CairoMakie.ylims!(ax, min_y, max_y)

    axislegend(; position= :rt, labelsize= 16)

    save(joinpath(savedir,"spectra_test_$(inchannel).png"), fig, px_per_unit = 2)

end
