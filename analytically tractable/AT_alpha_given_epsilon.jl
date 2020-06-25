push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# push!(LOAD_PATH, "/homes/crucinio/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
# custom packages
using smcems;
using wgf;
using diagnostics;

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
h(x) = pdf.(Normal(0.5, sqrt(sigmaH)), x);
g(x, y) = pdf.(Normal(x, sqrt(sigmaG)), y);

# common parameters
# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 1000;
# number of particles
Nparticles = 1000;
KDEx = range(0, stop = 1, length = 1000);

epsilon = 0.001;
Nrep = 10;
entropy  = zeros(Nrep, 1);
Threads.@threads for i=1:Nrep
    x0 = rand(1, Nparticles);
    xSMC, W = smc_AT_approximated_potential(Nparticles, Niter, epsilon, x0, M);
    # kde
    bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
    KDEySMC = weightedKDE(xSMC[end, :], W[end, :], bw, KDEx);
    _, _, _, _, entropy[i] = diagnosticsF(f, KDEx, KDEySMC);
end
target_entropy = mean(entropy);

interval = [0 0.5];
threshold = 10^-2;
resWGF = AT_alpha_WGF(target_entropy, interval, threshold, dt, Niter, Nparticles, "unif", M);

Nrep = 10;
entropyWGF  = zeros(Nrep, 1);
Threads.@threads for i=1:Nrep
    ### WGF
    x0 = rand(1)*ones(1, Nparticles);
    xWGF, _ =  wgf_AT(Nparticles, dt, Niter, resWGF[2], x0, M);
    # KDE
    # optimal bandwidth Gaussian
    KDEyWGF =  KernelEstimator.kerneldensity(xWGF[end,:], xeval=KDEx, h=bwnormal(xWGF[end,:]));
    entropyWGF[i] = -mean(remove_non_finite.(KDEyWGF .* log.(KDEyWGF)));
end
mean(entropyWGF)
