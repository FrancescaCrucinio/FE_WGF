function RL(KDisc, muCounts, Niter, rho0)
    # get dimension of unknown function ρ
    M = length(rho0);
    # initialize vector to store the values of ρ
    rhoCounts = zeros(Niter, M);
    # initial distribution
    rhoCounts[1, :] = rho0;

    den = KDisc * rhoCounts[1, :];
    # compute the numerator of the EM iterative formula
    num = muCounts .* KDisc;
    for t=2:Niter
        # update the denominator
        den = KDisc * rhoCounts[t-1, :];
        # update ρ
        rhoCounts[t, :] = rhoCounts[t-1, :] .* transpose(sum(num./den, dims = 1));
    end
    return rhoCounts
end
