abstract type ArtifactMetaData end

Base.@kwdef struct Turbulence2D <: ArtifactMetaData
    filename = "moist2d_512x512.hdf5"
    dataname = "moist2d_nx512_ny512"
    url = "https://caltech.box.com/shared/static/7oht5betdza54sjftk93ni0bsldtrptl.hdf5"
end

Base.@kwdef struct Turbulence2DComplete <: ArtifactMetaData
    filename = "turbulence2d_nx512_ny512.hdf5"
    dataname = "turbulence2d_nx512_ny512."
    url = "https://caltech.box.com/shared/static/0golc3ynh76v0lnv25xk2dsnxvpgacav.hdf5"
end

Base.@kwdef struct KuramotoSivashinsky <: ArtifactMetaData
    filename = "kuramoto_sivashinsky.hdf5"
    dataname = "kuramoto_sivashinsky"
    url = "https://caltech.box.com/shared/static/5va3rbvp908da3xliglf7h8r0y971dwj.hdf5"
end
