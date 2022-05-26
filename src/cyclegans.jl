struct CycleGAN
    generator_AB
    generator_BA
    discriminator_AB
    discriminator_BA
end

@functor CycleGAN

function update_cyclegan(opt_gen, opt_discr, cycleGAN, real_A, real_B)
    loss = Dict()

    # We only update the generators
    ps = Flux.params(cycleGAN.generator_AB, cycleGAN.generator_BA)

    # We only update the generators' parameters
    loss["G_loss"], back = Zygote.pullback(ps) do
        # produce fake images
        fake_A = cycleGAN.generator_AB(real_B)
        fake_B = cycleGAN.generator_BA(real_A)

        # update the discriminators' parameters explicityly
        loss["D_loss"] = update_discriminator(opt_discr, cycleGAN, real_A, real_B, fake_A, fake_B)

        # adversarial loss for both generators, the parameters
        # of the discriminators are fixed in this step.
        D_A_fake = cycleGAN.discriminator_AB(fake_A)
        D_B_fake = cycleGAN.discriminator_BA(fake_B)
        loss_G_A = mse(D_A_fake, ones(size(D_A_fake)))
        loss_G_B = mse(D_B_fake, ones(size(D_B_fake)))

        # cycle consistency loss for generators
        cycle_B = cycleGAN.generator_BA(fake_A)
        cycle_A = cycleGAN.generator_AB(fake_B)
        cycle_B_loss = mae(real_B, cycle_B)
        cycle_A_loss = mae(real_A, cycle_A)

        calculate_loss_generator(loss_G_A, loss_G_B, cycle_A_loss, cycle_B_loss)
    end
    grads = back(1.0f0) # Jacobian-vector product
    update!(opt_gen, ps, grads) # apply gradients

    return loss
end

function update_discriminator(opt_discr, cycleGAN, real_A, real_B, fake_A, fake_B)
    # We only handle the discriminators
    ps = Flux.params(cycleGAN.discriminator_AB, cycleGAN.discriminator_BA)

    # Update only the discriminators' parameters
    loss, back = Zygote.pullback(ps) do
        # calculate loss for A
        D_A_real = cycleGAN.discriminator_AB(real_A)
        D_A_fake = cycleGAN.discriminator_AB(fake_A)
        D_A_real_loss = mse(D_A_real, ones(size(D_A_real)))
        D_A_fake_loss = mse(D_A_fake, zeros(size(D_A_fake)))
        D_A_loss = D_A_real_loss + D_A_fake_loss

        # calculate loss for B
        D_B_real = cycleGAN.discriminator_BA(real_B)
        D_B_fake = cycleGAN.discriminator_BA(fake_B)
        D_B_real_loss = mse(D_B_real, ones(size(D_B_real)))
        D_B_fake_loss = mse(D_B_fake, zeros(size(D_B_fake)))
        D_B_loss = D_B_real_loss + D_B_fake_loss

        calculate_loss_discriminator(D_A_loss, D_B_loss)
    end
    grads = back(1.0f0) # Jacobian-vector product
    update!(opt_discr, ps, grads) # apply gradients

    return loss
end

# We will take the gradient by hand, aka we will use Flux.pullback, so we
# need to make sure we don't take the gradient of the loss function accidentally.
Zygote.@nograd update_discriminator

function calculate_loss_discriminator(real_loss, fake_loss)
    return (real_loss + fake_loss) / 2
end

function calculate_loss_generator(loss_G_A, loss_G_B, cycle_A_loss, cycle_B_loss; λ=10)
    return loss_G_A + loss_G_B + oftype(cycle_A_loss / 1, λ) * (cycle_A_loss + cycle_B_loss)
end
