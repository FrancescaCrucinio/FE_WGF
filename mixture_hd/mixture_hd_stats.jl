function mixture_hd_stats(x, W, component)
    # mean
    m = sum(W.*x[component, :])/sum(W);
    # variance
    v = sum(W.*x[component, :].^2)/sum(W) - m^2;
    # probability
    indices = prod((0 .<= x .<= 0.5), dims = 1);
    p = sum(W[indices[:]]);
    # grid on support of 1d marginal
    KDEx = range(0, stop = 1, length = 100);
    # W1 distance
    pi_marginal = MixtureModel(Normal, [(0.3, 0.07^2), (0.7, 0.1^2)], [1/3, 2/3]);
    w1 = (KDEx[2] - KDEx[1])*sum(abs.(quantile(rand(pi_marginal, 10^6), KDEx) .- quantile(x[component, :], weights(W), KDEx)));
    # KS distance
    true_cdf = cdf.(Normal(0.3, 0.07^2), KDEx)/3 .+ 2*cdf.(Normal(0.7, 0.1^2), KDEx)/3;
    empirical_cdf = ecdf(x[component, :], weights = W);
    ks = maximum(abs.(true_cdf .- empirical_cdf(KDEx)));

    return m, v, p, ks, w1
end
