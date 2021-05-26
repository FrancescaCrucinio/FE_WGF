function mixture_hd_stats(x, W, component)
    # mean
    m = sum(W.*x[component, :])/sum(W);
    # variance
    v = sum(W.*x[component, :].^2)/sum(W) - m^2;
    # probability
    indices = prod((0 .<= x .<= 0.5), dims = 1);
    p = sum(W[indices[:]]);

    return m, v, p
end
