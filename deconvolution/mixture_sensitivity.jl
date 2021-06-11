push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using RCall;
@rimport ks as rks
# custom packages
using wgf_prior;
using samplers;
# set seed
Random.seed!(1234);

# data for gaussian mixture example
pi(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
mu(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
K(x, y) = pdf.(Normal(x, 0.045), y);
sdK = 0.045;

function psi(piSample, a, m0, sigma0, muSample)
    loglik = zeros(1, length(muSample));
    for i=1:length(muSample)
        loglik[i] = mean(K.(piSample, muSample[i]));
    end
    loglik = -log.(loglik);
    kl = mean(loglik);
    prior = pdf.(Normal(m0, sigma0), piSample);
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    kl_prior = mean(log.(pihat./prior));
    return kl+a*kl_prior;
end

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 100;
# number of particles
Nparticles = 100;
# regularisation parameters
alpha = range(0, stop = 0.001, length = 100);

Nrep = 100;
E = zeros(length(alpha), Nrep);
for i=1:length(alpha)
    for j=1:Nrep
    muSample = Ysample_gaussian_mixture(10^3);
    x0 = sample(muSample, Nparticles, replace = !(Nparticles <= 10^3));
    # prior mean = mean of μ
    m0 = mean(muSample);
    sigma0 = std(muSample);
    # size of sample from μ
    M = min(Nparticles, length(muSample));
    # WGF
    x = wgf_DKDE_tamed(Nparticles, dt, Niter, alpha[i], x0, m0, sigma0, muSample, M, sdK);
    # estimate functional
    E[i, j] = psi(x[Niter, :], alpha[i], m0, sigma0, muSample);
    println("$i, $j")
    end
end

Eavg = mean(E, dims = 2);
p = plot(alpha, Eavg, lw = 2, label = "", legendfontsize = 15, tickfontsize = 10)
# savefig(p,"mixture_sensitivity_E.pdf")
