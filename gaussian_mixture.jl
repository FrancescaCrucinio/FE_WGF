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
f(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
h(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
g(x, y) = pdf.(Normal(x, 0.045, y));

# number of iterations
Niter = trunc(Int, 1e03);
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
lambda = 20;


x0 = rand(1, Nparticles);
# run WGF
x, drift =  wgf_gaussian_mixture(Nparticles, Niter, lambda, x0, M);
KDEyWGF = kerneldensity(x[end, :], xeval = KDEx);
stats = diagnosticsF(f, KDEx, KDEyWGF);

plot(f, 0, 1, lw = 3)
plot!(KDEx, KDEyWGF, lw = 3)
