#= One step late EM for penalised likelihood
OUTPUTS
1 - estimate of π at bin centres
2 - value of functional
INPUTS
'muDisc' discretisation of μ
'sigmaK' standard deviation of mixing normal
'alpha' regularisation parameter
'Niter' number of iterations
'pi0' reference measure evaluated at bin centres
'pi_init' initial distribution
'KDEeval' bin centres
'funtional' if true the value of the functional at each iteration is returned
=#

function osl_em(muDisc, sigmaK, alpha, Niter, pi0, pi_init, KDEeval, functional)
    # get dimension of unknown function f
    M = length(pi_init);
    # initial distribution
    pi = copy(pi_init);
    # discretisation grid
    dx = KDEeval[2, 1] .- KDEeval[1, 1];
    # dimension
    d = size(KDEeval, 2);
    # value of functional
    E = zeros(Niter);
    if(functional)
        prior_kl = dx^d * sum(pi .* log.(pi./pi0));
        den = zeros(1, M);
        for b=1:M
            den[b] = (pi' * pdf(MvNormal(KDEeval[b, :], sigmaK^2*Matrix{Float64}(I, d, d)), KDEeval'))[1];
        end
        kl = dx^d * sum(muDisc .* log.(den));
        E[1] = kl + alpha * prior_kl;
    end
    for n=2:Niter
        # update the denominator
        den = zeros(1, M);
        for b=1:M
            den[b] = (pi' * pdf(MvNormal(KDEeval[b, :], sigmaK^2*Matrix{Float64}(I, d, d)), KDEeval'))[1];
        end
        for b=1:M
            # numerator + penalty
            pi[b] = pi[b]*sum(muDisc .* pdf(MvNormal(KDEeval[b, :], sigmaK^2*Matrix{Float64}(I, d, d)), KDEeval')./den')/
                (1 + alpha + alpha*(log(pi[b]/pi0[b])));
        end
        pi[pi.<0] .= 0;
        if(functional)
            prior_kl = dx^d * sum(pi .* log.(pi./pi0));
            kl = dx^d * sum(muDisc .* log.(den));
            E[n] = kl + alpha * prior_kl;
        end
    end
    return pi, E
end
