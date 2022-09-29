"""
Helper to make an animation from a batch of images.
"""
function convert_to_animation(x, hpdata)
    frames = size(x)[end]
    batches = size(x)[end-1]
    animation = @animate for i = 1:frames+frames÷4
        if i <= frames
            heatmap(
                convert_to_image(x[:, :, :, :, i], hpdata.inchannels),
                title="Iteration: $i out of $frames"
            )
        else
            heatmap(
                convert_to_image(x[:, :, :, :, end], hpdata.inchannels),
                title="Iteration: $frames out of $frames"
            )
        end
    end
    return animation
end

function plot_result(model, save_path, hpdata; num_images=25, num_steps=500)
    device = gpu
    @info "Using device: $device"
    model = model |> device
    time_steps, Δt, init_x = setup_sampler(model, device, hpdata; num_images = num_images, num_steps = num_steps)
    # Euler-Maruyama
    euler_maruyama = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
    sampled_noise = convert_to_image(init_x, hpdata.inchannels)
    Images.save(joinpath(save_path, "sampled_noise.jpeg"), sampled_noise)
    em_images = convert_to_image( euler_maruyama, hpdata.inchannels)
    Images.save(joinpath(save_path, "em_images.jpeg"), em_images)

    # Predictor Corrector
    pc = predictor_corrector_sampler(model, init_x, time_steps, Δt)
    pc_images = convert_to_image(pc, hpdata.inchannels)
    Images.save(joinpath(save_path, "pc_images.jpeg"), pc_images)
end

function convert_to_image(x::AbstractArray{T,N}, inchannels) where {T,N}
    ysize = size(x)[end]
    if inchannels == 1
        x =  Gray.(permutedims(vcat(reshape.(Flux.chunk(x |> cpu, ysize), 32, :)...), (2, 1)))
        return x
    elseif inchannels ==3
        tmp = reshape.(Flux.chunk(permutedims(x, (3,2,1,4)) |> cpu, ysize), 3, 32, :)
        rgb = colorview.(Ref(RGB), tmp)
        return vcat(rgb...)
    else
        error("Number of inchannels not supported")
    end
    return x
end 
