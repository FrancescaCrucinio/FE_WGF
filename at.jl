push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
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
using diagnostics;
using smcems;
using wgf;
using samplers;

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
h(x) = pdf.(Normal(0.5, sqrt(sigmaH)), x);
g(x, y) = pdf.(Normal(x, sqrt(sigmaG)), y);

# dt and final time
dt = 1e-03;
T = 1;

# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
lambda = 25;


x0 = rand(1, Nparticles);
# run WGF
x, drift =  wgf_AT_approximated(Nparticles, Niter, lambda, x0, M);

KDEyWGF = kerneldensity(x[end, :], xeval = KDEx);
stats = diagnosticsF(f, KDEx, KDEyWGF);

p = StatsPlots.plot(f, 0, 1, lw = 3, label = "True f")
StatsPlots.plot!(KDEx, KDEyWGF, lw = 3, label = "WGF")

savefig(p, "at.pdf")
