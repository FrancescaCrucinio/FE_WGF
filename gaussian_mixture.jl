push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
using LaTeXStrings;
# custom packages
using diagnostics;
using wgf;

pyplot()
# set seed
Random.seed!(1234);

# data for gaussian mixture example
f(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
h(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
g(x, y) = pdf.(Normal(x, 0.045, y));

# dt and number of iterations
dt = 1e-03;
Niter = 100;

# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
lambda = 0.001;


x0 = 0.5*ones(1, Nparticles);
# run WGF
x, _ =  wgf_gaussian_mixture(Nparticles, dt, Niter, lambda, x0, M);

# KDE
KDEyWGF = KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=bwnormal(x[end,:]));
stats = diagnosticsF(f, KDEx, KDEyWGF);

p = StatsPlots.plot(f, 0, 1, lw = 3, label = L"True $\rho$", color=:black,
    legendfontsize = 10);
StatsPlots.plot!(KDEx, KDEyWGF, lw = 3, label = "WGF", color = 1);

savefig(p, "mixture.pdf")
