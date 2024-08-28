using FFTW
using BSON
using CliMAgen
using Flux    
using TOML
using HDF5
using Plots
"""
batch_spectra(data)

Computes and returns the mean azimuthally averaged power 
spectrum for the data, where the mean is taken
over the batch dimension,
but not over the channel dimension.
"""
function batch_spectra(data)
    statistics = x -> hcat(power_spectrum2d(x)...)
    data = data |> Flux.cpu
    results = mapslices(statistics, data, dims=[1, 2])
    k = results[:, 2, 1, 1]
    results = results[:, 1, :, :]
    spectrum = mean(results, dims=3)
    return spectrum, k
end

function power_spectrum2d(img)
    @assert size(img)[1] == size(img)[2]
    dim = size(img)[1]
    img_fft = abs.(fft(img .- mean(img)))
    m = Array(img_fft / size(img_fft, 1)^2)
    if mod(dim, 2) == 0
        rx = range(0, stop=dim - 1, step=1) .- dim / 2 .+ 1
        ry = range(0, stop=dim - 1, step=1) .- dim / 2 .+ 1
        R_x = circshift(rx', (1, dim / 2 + 1))
        R_y = circshift(ry', (1, dim / 2 + 1))
        k_nyq = dim / 2
    else
        rx = range(0, stop=dim - 1, step=1) .- (dim - 1) / 2
        ry = range(0, stop=dim - 1, step=1) .- (dim - 1) / 2
        R_x = circshift(rx', (1, (dim + 1) / 2))
        R_y = circshift(ry', (1, (dim + 1) / 2))
        k_nyq = (dim - 1) / 2
    end
    r = zeros(size(rx, 1), size(ry, 1))
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        r[i, j] = sqrt(R_x[i]^2 + R_y[j]^2)
    end
    k = range(1, stop=k_nyq, step=1)
    endk = size(k, 1)
    contribution = zeros(endk)
    spectrum = zeros(endk)
    for N in 2:Int64(k_nyq - 1)
        for i in 1:size(rx, 1), j in 1:size(ry, 1)
            if (r[i, j] <= (k'[N+1] + k'[N]) / 2) &&
            (r[i, j] > (k'[N] + k'[N-1]) / 2)
                spectrum[N] =
                    spectrum[N] + m[i, j]^2
                contribution[N] = contribution[N] + 1
            end
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j] <= (k'[2] + k'[1]) / 2)
            spectrum[1] =
                spectrum[1] + m[i, j]^2
            contribution[1] = contribution[1] + 1
        end
    end
    for i in 1:size(rx, 1), j in 1:size(ry, 1)
        if (r[i, j] <= k'[endk]) &&
        (r[i, j] > (k'[endk] + k'[endk-1]) / 2)
            spectrum[endk] =
                spectrum[endk] + m[i, j]^2
            contribution[endk] = contribution[endk] + 1
        end
    end
    spectrum = spectrum ./ contribution

    return spectrum, k
end

"""
t_cutoff(power::FT, k::FT, N::FT, σ_max::FT, σ_min::FT) where {FT}

Computes and returns the time `t` at which the power of 
the radially averaged Fourier spectrum of white noise of size NxN, 
with variance σ_min^2(σ_max/σ_min)^(2t), at wavenumber `k`,
is equal to `power`.
"""
function t_cutoff(power::FT, k::FT, N::FT, σ_max::FT, σ_min::FT) where {FT}
    return 1/2*log(power*N^2/σ_min^2)/log(σ_max/σ_min)
end


package_dir = pkgdir(CliMAgen)
include(joinpath(package_dir,"examples/conus404/preprocessing_utils.jl"))
include(joinpath(package_dir,"examples/utils_data.jl")) # for data loading
FT = Float32
experiment_toml = "Experiment_tmax_standard_scaling.toml"
# read experiment parameters from file
params = TOML.parsefile(experiment_toml)
params = CliMAgen.dict2nt(params)
# unpack params
savedir = params.experiment.savedir
samplesdir = joinpath(savedir, "downscaling")
rngseed = params.experiment.rngseed
nogpu = params.experiment.nogpu
standard_scaling = params.data.standard_scaling
fname_train = params.data.fname_train
fname_test = params.data.fname_test
precip_channel = params.data.precip_channel
precip_floor::FT = params.data.precip_floor
low_pass = params.data.low_pass
low_pass_k = params.data.low_pass_k
n_pixels = params.data.n_pixels
inchannels = params.model.inchannels

if params.downscaling.coarse_res_data_file == "nothing"
    # get train and test data (hi-res data)
    (_, xtest) = get_raw_data_conus404(fname_train, fname_test, precip_channel; precip_floor = precip_floor, FT=FT)
    # Filter to create lo-res standin
    wn=8
    xsource = lowpass_filter(xtest, wn)
    # Preprocess the coarse resolution data
else
    # get test data (already low-res!) and make preprocessing parameters
    @info "using real coarse resolution data"
    fid = HDF5.h5open(params.downscaling.coarse_res_data_file, "r")
    xsource = HDF5.read(fid["coarse_res_data"])
    close(fid)
end
# Determine the turnaround time for the diffusion bridge,
# corresponding to the power spectral density at which
# the two sets of images begin to differ from each other
source_spectra, k = batch_spectra(xsource[:,:,:,1:20])

(xtarget,_) = get_raw_data_conus404(fname_train, fname_test, precip_channel; precip_floor = precip_floor, FT=FT)

target_spectra, k = batch_spectra(xtarget[:,:,:,1:20])
N = FT(size(xsource)[1])
cutoff_idx =3 # manually determined

checkpoint_path = joinpath(savedir, "checkpoint.bson")
BSON.@load checkpoint_path model model_smooth opt opt_smooth
model = model_smooth


if cutoff_idx > 0
    k_cutoff = FT(k[cutoff_idx])
    source_power_at_cutoff = FT(mean(source_spectra[cutoff_idx,:,:]))
    target_power_at_cutoff = FT(mean(target_spectra[cutoff_idx,:,:]))
    reverse_t_end = FT(t_cutoff(target_power_at_cutoff, k_cutoff, N, model.σ_max, model.σ_min))
    forward_t_end = FT(t_cutoff(source_power_at_cutoff, k_cutoff, N, model.σ_max, model.σ_min))
else
    reverse_t_end = FT(1)
    forward_t_end = FT(1)
end
@show forward_t_end
@show reverse_t_end


