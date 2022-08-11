using BSON: @save, @load
using CUDA
using Dates
using Flux
using Flux: params, update!
using FluxTraining
using HDF5
using MLUtils
using Statistics: mean

using Downscaling: PatchDiscriminator, UNetGenerator

FT = Float32

# Parameters
Base.@kwdef struct HyperParams
    λ = FT(10.0)
    λid = FT(5.0)
end

function get_dataloader(path; field="vorticity", split_ratio=0.5, batch_size=1)
    fid = h5open(path, "r")
    X_lo_res = read(fid, "low_resolution/" * field)
    X_hi_res = read(fid, "high_resolution/" * field)
    close(fid)

    # TODO: needs to be handled by a data transfomer object, e.g.
    # by using a MinMaxScaler object
    # normalize data
    X_lo_res .-= (maximum(X_lo_res) + minimum(X_lo_res)) / 2
    X_lo_res ./= (maximum(X_lo_res) - minimum(X_lo_res)) / 2
    X_hi_res .-= (maximum(X_hi_res) + minimum(X_hi_res)) / 2
    X_hi_res ./= (maximum(X_hi_res) - minimum(X_hi_res)) / 2

    # fix data types to Float32 for training
    X_lo_res = FT.(X_lo_res)
    X_hi_res = FT.(X_hi_res)

    data_training, data_validation = MLUtils.splitobs((X_lo_res, X_hi_res), at=split_ratio)
    loader_training = Flux.DataLoader(data_training, batchsize=batch_size, shuffle=true)
    loader_validation = Flux.DataLoader(data_validation, batchsize=batch_size, shuffle=true)

    return (training=loader_training, validation=loader_validation,)
end

function gen_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams)
    lores_fake = gen_lores(hires) # Fake image generated in lores domain
    hires_fake = gen_hires(lores) # Fake image generated in hires domain

    hires_fake_prob = dscr_hires(hires_fake) # Probability that generated image in hires domain is real
    lores_fake_prob = dscr_lores(lores_fake) # Probability that generated image in lores domain is real

    gen_lores_loss = mean((lores_fake_prob .- 1) .^ 2)
    rec_lores_loss = mean(abs.(hires - gen_hires(lores_fake))) # Cycle-consistency loss for hires domain
    idt_lores_loss = mean(abs.(gen_hires(hires) .- hires)) # Identity loss for hires domain
    gen_hires_loss = mean((hires_fake_prob .- 1) .^ 2)
    rec_hires_loss = mean(abs.(lores - gen_lores(hires_fake))) # Cycle-consistency loss for lores domain
    idt_hires_loss = mean(abs.(gen_lores(lores) .- lores)) # Identity loss for lores domain

    return gen_lores_loss + gen_hires_loss + hparams.λ * (rec_lores_loss + rec_hires_loss) + hparams.λid * (idt_lores_loss + idt_hires_loss)
end

function dscr_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires)
    lores_fake = gen_lores(hires) # Fake image generated in lores domain
    hires_fake = gen_hires(lores) # Fake image generated in hires domain

    lores_fake_prob = dscr_lores(lores_fake) # Probability that generated image in lores domain is real
    lores_real_prob = dscr_lores(lores) # Probability that an original image in lores domain is real
    hires_fake_prob = dscr_hires(hires_fake) # Probability that generated image in hires domain is real
    hires_real_prob = dscr_hires(hires) # Probability that an original image in hires domain is real

    real_lores_loss = mean((lores_real_prob .- 1) .^ 2)
    fake_lores_loss = mean((lores_fake_prob .- 0) .^ 2)
    real_hires_loss = mean((hires_real_prob .- 1) .^ 2)
    fake_hires_loss = mean((hires_fake_prob .- 0) .^ 2)

    return real_lores_loss + fake_lores_loss + real_hires_loss + fake_hires_loss
end

function train_step!(opt_gen, opt_dscr, gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams)
    # Optimize Discriminators
    ps = params(params(dscr_lores)..., params(dscr_hires)...)
    gs = gradient(() -> dscr_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires), ps)
    update!(opt_dscr, ps, gs)

    # Optimize Generators
    ps = params(params(gen_hires)..., params(gen_lores)...)
    gs = gradient(() -> gen_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams), ps)
    update!(opt_gen, ps, gs)
end

function fit!(opt_gen, opt_dscr, gen_lores, gen_hires, dscr_lores, dscr_hires, data, hparams, nepochs)
    # Training loop
    @info "Training begins..."
    for epoch in 1:nepochs
        epoch_start = Dates.now()
        @info "Epoch: $epoch -------------------------------------------------------------------"
        for (lores, hires) in data
            train_step!(opt_gen, opt_dscr, gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams)
        end

        # print current error
        g_loss = gen_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires, hparams)
        d_loss = dscr_loss(gen_lores, gen_hires, dscr_lores, dscr_hires, lores, hires)
        @info "Epoch: $epoch - Generator loss: $g_loss, Discriminator loss: $d_loss"

        # store current model
        @info "Checkpointing model."
        output_path = joinpath(@__DIR__, "/output/checkpoint_latest.bson")
        networks_cpu = (gen_lores, gen_hires, dscr_lores, dscr_hires) |> cpu
        @save output_path networks_cpu

        # print time elapsed
        @info "Epoch duration: $(Dates.canonicalize(Dates.now() - epoch_start))"
    end
end

function train(; cuda=true, lr=FT(0.0002), nepochs=100)
    if cuda && CUDA.has_cuda()
        device = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # hyper params
    hparams = HyperParams()

    # data
    data = get_dataloader("../../data/moist2d/moist2d_512x512.hdf5", split_ratio=0.5, batch_size=1)
    data = data.training

    # models
    nchannels = 1
    gen_hires = UNetGenerator(nchannels) |> device # Generator For lores->hires
    gen_lores = UNetGenerator(nchannels) |> device # Generator For hires->lores
    dscr_hires = PatchDiscriminator(nchannels) |> device # Discriminator For lores domain
    dscr_lores = PatchDiscriminator(nchannels) |> device # Discriminator For hires domain

    # optimizers
    opt_gen = ADAM(lr, (0.5, 0.999))
    opt_dis = ADAM(lr, (0.5, 0.999))

    fit!(opt_gen, opt_dis, gen_lores, gen_hires, dscr_lores, dscr_hires, data, hparams, nepochs)
end

function get_model()
    model_path = joinpath(@__DIR__, "../model/")
    model_file = readdir(model_path)[end]

    return BSON.load(joinpath(model_path, model_file), @__MODULE__)[:model]
end

train()
