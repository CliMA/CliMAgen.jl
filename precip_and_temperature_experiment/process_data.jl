using Random

function ensemble_average_context(field; N = 251, contextS = 1)
    ML, NL, S, ET = size(field)
    r_field = reshape(field, ML, NL, S, N, 45)
    tmp = mean(r_field, dims = 5)
    tmp_field = zeros(Float32, ML, NL, contextS, N, 45)
    for i in 1:N
        for j in 1:contextS
            tmp_field[:, :, j, i, :] .=  tmp[:, :, j, i, :]
        end
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96, contextS, N*45)

    return cat(field, reshaped_tmp_field, dims = 3)
end

function ensemble_spatial_average_context(field; N = 251, contextS = 1)
    ML, NL, S, ET = size(field)
    r_field = reshape(field, ML, NL, S, N, 45)
    tmp = mean(r_field, dims = (1, 2, 5))
    tmp_field = zeros(Float32, ML, NL, contextS, N, 45)
    for i in 1:N
        for j in 1:contextS
            tmp_field[:, :, j, i, :] .=  tmp[:, :, j, i, :]
        end
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96, contextS, N*45)

    return cat(field, reshaped_tmp_field, dims = 3)
end


#=
import CliMAgen: GaussianFourierProjection
function GaussianFourierProjection(embed_dim::Int, embed_dim2::Int, scale::FT) where {FT}
    Random.seed!(1234)
    W = randn(FT, embed_dim ÷ 2, embed_dim2) .* scale
    return GaussianFourierProjection(W)
end
gfp = GaussianFourierProjection(192, 96, 30f0) # 30.0f0 is the default

# gaussian fourier projection
function gmt_embedding(field, tas_rescaled, gfp)
    tmp_field = zeros(Float32, 192, 96, 251, 45)
    for i in 1:251
        tembedding = reshape(gfp(tas_rescaled[i]), (192, 96, 1))
        tmp_field[:, :, i, :] .=  tembedding * gfp_scale
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96,1, 251*45)

    return cat(field, reshaped_tmp_field, dims = 3)
end

# ensemble mean
function gmt_embedding_2(field, tas_rescaled, gfp; N = 251)
    tmp_field = zeros(Float32, 192, 96, N, 45)
    r_field = reshape(field, 192, 96, N, 45)
    for i in 1:N
        tmp_field[:, :, i, :] .= mean(r_field, dims = 4)[:, :, i, :]
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96,1, N*45)

    return cat(field, reshaped_tmp_field, dims = 3)
end

# spatial average
function gmt_embedding_3(field, tas_rescaled, gfp; N = 251)
    tmp_field = zeros(Float32, 192, 96, N, 45)
    r_field = reshape(field, 192, 96, N, 45)
    field1 = mean(r_field, dims = 4)[:, :, 1, :]
    fieldend = mean(r_field, dims = 4)[:, :, 151, :]
    tmp = mean(r_field, dims = (1,2, 4))
    for i in 1:N
        τ = 1 - (i-1) / (N-1)
        tmp_field[:, :, i, :] .=  tmp[:,:,i,:] # field1 * τ + fieldend * (1 - τ)
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96,1, N*45)

    return cat(field, reshaped_tmp_field, dims = 3)
end

# zonal average
function gmt_embedding_4(field, tas_rescaled, gfp; N = 251)
    tmp_field = zeros(Float32, 192, 96, N, 45)
    r_field = reshape(field, 192, 96, N, 45)
    field1 = mean(r_field, dims = 4)[:, :, 1, :]
    fieldend = mean(r_field, dims = 4)[:, :, 151, :]
    tmp = mean(r_field, dims = (1, 4))
    for i in 1:N
        τ = 1 - (i-1) / (N-1)
        tmp_field[:, :, i, :] .=  tmp[:,:,i,:] # field1 * τ + fieldend * (1 - τ)
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96,1, N*45)

    return cat(field, reshaped_tmp_field, dims = 3)
end

# greedy pattern scaling
function gmt_embedding_5(field, tas_rescaled, gfp; N = 251)
    tmp_field = zeros(Float32, 192, 96, N, 45)
    r_field = reshape(field, 192, 96, N, 45)
    field1 = mean(r_field, dims = 4)[:, :, 1, :]
    indmax = 151
    fieldend = mean(r_field, dims = 4)[:, :, indmax, :]
    pretau = mean(r_field, dims = (1,2, 4))[1,1,:,1]
    taumax = maximum(pretau[1:indmax])
    taumin = minimum(pretau[1:indmax])
    tau = (taumax .- pretau) ./ (taumax .- taumin)
    for i in 1:N
        τ = 1-tau[i]
        tmp_field[:, :, i, :] .=  field1 * τ + fieldend * (1 - τ)
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96,1, N*45)

    return cat(field, reshaped_tmp_field, dims = 3)
end

## temperature conditional information
# spatial average
function gmt_embedding_6(field, tas_rescaled, gfp; N = 251)
    hfile = h5open("/nobackup1/users/sandre/GaussianEarth/tas_field_month_1.hdf5", "r")
    temperaturefield = FT.(read(hfile["timeseries"]) ) 
    close(hfile)
    inds = 1:N
    rtemperaturefield = reshape(temperaturefield, (192, 96, 251, 45))[:, :, inds, :]
    tmp_field = zeros(Float32, 192, 96, N, 45)
    tmp = mean(rtemperaturefield, dims = (1,2, 4))
    for i in inds
        tmp_field[:, :, i, :] .=  tmp[:,:, i, :]
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96,1, N*45)
    return cat(field, reshaped_tmp_field, dims = 3)
end

# ensemble average
function gmt_embedding_7(field, tas_rescaled, gfp; N = 251)
    hfile = h5open("/nobackup1/users/sandre/GaussianEarth/tas_field_month_1.hdf5", "r")
    temperaturefield = FT.(read(hfile["timeseries"])) 
    close(hfile)
    inds = 1:N
    rtemperaturefield = reshape(temperaturefield, (192, 96, 251, 45))[:, :, inds, :]
    tmp_field = zeros(Float32, 192, 96, N, 45)
    tmp = mean(rtemperaturefield, dims = 4)
    for i in inds
        tmp_field[:, :, i, :] .=  tmp[:, :, i, :]
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96,1, N*45)
    return cat(field, reshaped_tmp_field, dims = 3)
end

function gmt_embedding_8(field, tas_rescaled, gfp; N = 251)
    hfile = h5open("/nobackup1/users/sandre/GaussianEarth/tas_field_month_1.hdf5", "r")
    temperaturefield = FT.(read(hfile["timeseries"])) 
    close(hfile)
    inds = 1:N
    rtemperaturefield = reshape(temperaturefield, (192, 96, 251, 45))[:, :, inds, :]
    tmp_field = zeros(Float32, 192, 96, N, 45)
    r_field = reshape(field, 192, 96, N, 45)
    field1 = mean(rtemperaturefield, dims = 4)[:, :, 1, :]
    indmax = 151
    fieldend = mean(rtemperaturefield, dims = 4)[:, :, indmax, :]
    pretau = mean(rtemperaturefield, dims = (1,2, 4))[1,1,:,1]
    taumax = maximum(pretau[1:indmax])
    taumin = minimum(pretau[1:indmax])
    tau = (taumax .- pretau) ./ (taumax .- taumin)
    for i in 1:N
        τ = 1-tau[i]
        tmp_field[:, :, i, :] .=  field1 * τ + fieldend * (1 - τ)
    end
    reshaped_tmp_field = reshape(tmp_field, 192, 96,1, N*45)

    return cat(field, reshaped_tmp_field, dims = 3)
end
=#