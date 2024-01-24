function bootstrap(metric, samples, likelihood_ratio, nboot)
    observable_lr_product = metric.(samples) .* likelihood_ratio
    output = zeros(nboot)
    nsamples = length(samples)
    indices = 1:1:nsamples
    random_indices = zeros(Int, nsamples)
    for i in 1:nboot
        random_indices .= StatsBase.sample(indices, nsamples, replace=true)
        output[i] = mean(observable_lr_product[random_indices]) # takes expectation
    end
    return output
end
