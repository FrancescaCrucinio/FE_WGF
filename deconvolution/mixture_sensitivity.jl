# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using wgf_prior;
using samplers;

# set seed
Random.seed!(1234);

# data for gaussian mixture example
rho(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
mu(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
K(x, y) = pdf.(Normal(x, 0.045), y);
sdK = 0.045;

# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);



# functional approximation
function psiWGF(piSample, a, m0, sigma0, muSample)
	# build KDE
	Rpihat = rks.kde(x = piSample, var"eval.points" = KDEx);
    pihat = abs.(rcopy(Rpihat[3]));
	# kl
	trueMu = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), KDEx)/3 +
			pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), KDEx)/3;
	# approximated value
	delta = KDEx[2] - KDEx[1];
	hatMu = zeros(1, length(KDEx));
	# convolution with approximated f
	# this gives the approximated value
	for i=1:length(KDEx)
		hatMu[i] = delta*sum(pdf.(Normal.(KDEx, sdK), KDEx[i]).*pihat);
	end
	kl = kl_divergence(trueMu, hatMu);
    # regularization term
    prior = pdf.(Normal(m0, sigma0), KDEx);
    kl_prior = kl_divergence(pihat, prior);
    return kl+a*kl_prior;
end

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 200;
# number of particles
Nparticles = 500;
# regularisation parameters
alpha = range(0, stop = 0.1, length = 10);
L = 10;
EWGF = zeros(length(alpha), L);

for i=1:length(alpha)
    for l=1:L
        muSample = Ysample_gaussian_mixture(10^3);
        x0 = sample(muSample, Nparticles, replace = !(Nparticles <= 10^3));
        # prior mean = mean of μ
        m0 = mean(muSample);
        sigma0 = std(muSample);
        # size of sample from μ
        M = min(Nparticles, length(muSample));
        # WGF
        xWGF = wgf_DKDE_tamed(Nparticles, dt, Niter, alpha[i], x0, m0, sigma0, muSample, M, sdK);
        # estimate functional
        EWGF[i, l] = psiWGF(xWGF[Niter, :], alpha[i], m0, sigma0, muSample);
        println("$i, $l")
    end
end
p = plot(alpha,  mean(EWGF, dims = 2), lw = 3, tickfontsize = 15)
# savefig(p,"mixture_sensitivity_E.pdf")
