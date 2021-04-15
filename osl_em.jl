#= One step late EM for penalised likelihood
OUTPUTS
1 - estimate of π at bin centres
INPUTS
'muDisc' discretisation of μ
'sigmaK' standard deviation of mixing normal
'alpha' regularisation parameter
'Niter' number of iterations
'pi0' reference measure evaluated at bin centres
'pi_init' initial distribution
=#

function osl_em(muDisc, sigmaK, alpha, Niter, pi0, pi_init)
    # get dimension of unknown function f
    M = length(pi_init);
    # initial distribution
    pi = pi_init;

    for n=2:Niter
        # update the denominator
        den = zeros(1, M);
        for d=1:M
            den[d] = pi' * mvnpdf(eval, eval(d, :), sigmaG^2*eye(p));
        end
        for b=1:M
            # numerator + penalty
            pi[b] = pi[b]*sum(muDisc .* mvnpdf(eval(b, :), eval, sigmaG^2*eye(p))./den')/
                (1 + alpha + alpha*log(pi[b]/pi0[b]));
        end
    end
    return pi
end
