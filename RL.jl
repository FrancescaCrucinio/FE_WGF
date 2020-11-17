function RL(KDisc, muCounts, Niter, x0)
    # get dimension of unknown function f
    M = size(KDisc, 2);
    # initialize vector to store the values of f
    rhoCounts = zeros(Niter, M);
    # initial distribution
    rhoCounts[1, :] = x0;

    # compute the numerator of the EM iterative formula
    num = muCounts .* KDisc;
    for t=2:Niter
        # update the denominator
        den = KDisc * rhoCounts[t-1,:];
        # update f
        rhoCounts[t, :] = rhoCounts[t-1, :] .* transpose(sum(num./den, dims = 1));
    end
    return rhoCounts
end
