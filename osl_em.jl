function osl_em(gDisc, hDisc, Niter, x0, alpha)
    # get dimension of unknown function f
    M = size(gDisc, 2);
    # initialize vector to store the values of f
    f = zeros(Niter, M);
    # initial distribution
    f[1, :] = x0;

    # compute the numerator of the EM iterative formula
    num = hDisc .* gDisc;
    for t=2:Niter
        # update the denominator
        den = gDisc * f[t-1,:];
        # update f
        f[t, :] = f[t-1, :]./(1 .+ alpha*(1 .+ log.(f[t-1, :]))) .* transpose(sum(num./den, dims = 1));
    end
    return f
end
