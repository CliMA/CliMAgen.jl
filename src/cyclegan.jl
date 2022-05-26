struct Cyclegan
    generator_AB
    generator_BA
    discriminator_AB
    discriminator_BA
end

@functor Cyclegan

function train_discr(cycleGAN, original_A, original_B, fake_A, fake_B, opt_discr)
    ps = Flux.params(cycleGAN.discriminator_AB, cycleGAN.discriminator_BA)

    loss, back = Zygote.pullback(ps) do
        # calculate loss for A
        D_A_real = cycleGAN.discriminator_AB(original_A)
        D_A_fake = cycleGAN.discriminator_AB(fake_A)
        D_A_real_loss = mse(D_A_real, ones(size(D_A_real)))
        D_A_fake_loss = mse(D_A_fake, zeros(size(D_A_fake)))
        D_A_loss = D_A_real_loss + D_A_fake_loss

        # calculate loss for B
        D_B_real = cycleGAN.discriminator_BA(original_B)
        D_B_fake = cycleGAN.discriminator_BA(fake_B)
        D_B_real_loss = mse(D_B_real, ones(size(D_B_real)))
        D_B_fake_loss = mse(D_B_fake, zeros(size(D_B_fake)))
        D_B_loss = D_B_real_loss + D_B_fake_loss

        calculate_loss_discr(D_A_loss, D_B_loss)
    end

    grads = back(1.0f0)
    update!(opt_discr, ps, grads)

    return loss
end

Zygote.@nograd train_discr

function train_gan(cycleGAN, original_A, original_B, opt_gen, opt_discr)
    loss = Dict()
    ps = Flux.params(cycleGAN.generator_AB, cycleGAN.generator_BA)

    loss["G_loss"], back = Zygote.pullback(ps) do
        # produce fake images
        fake_A = cycleGAN.generator_AB(original_B)
        fake_B = cycleGAN.generator_BA(original_A)

        # update the discriminator loss
        loss["D_loss"] = train_discr(cycleGAN,
            original_A, original_B,
            fake_A, fake_B, opt_discr)

        # adversarial loss for both generators
        D_A_fake = cycleGAN.discriminator_AB(fake_A)
        D_B_fake = cycleGAN.discriminator_BA(fake_B)
        loss_G_A = mse(D_A_fake, ones(size(D_A_fake)))
        loss_G_B = mse(D_B_fake, ones(size(D_B_fake)))

        # cycle consistency loss
        cycle_B = cycleGAN.generator_BA(fake_A)
        cycle_A = cycleGAN.generator_AB(fake_B)
        cycle_B_loss = mae(original_B, cycle_B)
        cycle_A_loss = mae(original_A, cycle_A)

        # identity loss (remove these for efficiency if you set lambda_identity=0)
        #identity_B = gen_B(original_B)
        #identity_A = gen_A(original_A)
        #identity_B_loss = mae(original_B, identity_B)
        #identity_A_loss = mae(original_A, identity_A)

        calculate_loss_gen(loss_G_A, loss_G_B, cycle_A_loss, cycle_B_loss)
    end

    grads = back(1.0f0)
    update!(opt_gen, ps, grads)

    return loss
end

function calculate_loss_discr(real, fake)
    return (real + fake) / 2
end

function calculate_loss_gen(loss_G_A, loss_G_B, cycle_A_loss,
    cycle_B_loss)
    return loss_G_A
    +loss_G_B
    +cycle_A_loss * 10
    +cycle_B_loss * 10
end
