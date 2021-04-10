push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using wgf_prior;

# Compare initial distributions

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaK = 0.045^2;
sigmaPi = 0.043^2;
sigmaMu = sigmaPi + sigmaK;
pi(x) = pdf.(Normal(0.5, sqrt(sigmaPi)), x);
mu(x) = pdf.(Normal(0.5, sqrt(sigmaMu)), x);
K(x, y) = pdf.(Normal(x, sqrt(sigmaK)), y);

# functional approximation
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

# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 100;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 500;
# regularisation parameter
alpha = 0.05;

# initial distributions
x0 = 0.5 * ones(1, Nparticles);
# sample from μ
muSample = rand(Normal(0.5, sqrt(sigmaMu)), 10^4);
m0 = mean(muSample);
sigma0 = std(muSample);
# size of sample from μ
M = min(Nparticles, length(muSample));

reference = ["normal" "uniform"];
E = zeros(Niter, length(reference));
for i=1:length(reference)
    x = wgf_AT_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M, reference[i]);
    # estimate functional
    E[i] = psi(x[Niter, :], alpha, m0, sigma0, muSample);
end
