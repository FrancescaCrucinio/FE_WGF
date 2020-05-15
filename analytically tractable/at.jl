# push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
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
using wgf;

# Plot AT example and exact minimiser

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
T = 10;
Niter = trunc(Int, 1/dt);
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(-0, stop = 1, length = 1000);
# number of particles
Nparticles = 10000;
# regularisation parameter
lambda = 0.025;


x0 = rand(1, Nparticles);
### WGF
x, _ =  wgf_AT(Nparticles, Niter, lambda, x0, M);
# KDE
# optimal bandwidth Gaussian
KDEyWGF1 =  KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=bwnormal(x[end,:]));
# bw = dt
KDEyWGF2 =  KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=dt);

### exact minimiser
variance, _  = AT_exact_minimiser(sigmaG, sigmaH, lambda);
ExactMinimiser(x) = pdf.(Normal(0.5, sqrt(variance)), x);

# plot
p = StatsPlots.plot(f, 0, 1, lw = 3, label = "True f")
StatsPlots.plot!(ExactMinimiser, 0, 1, lw = 3, label = "Exact minimiser")
StatsPlots.plot!(KDEx, KDEyWGF1, lw = 3, label = "WGF")
StatsPlots.plot!(KDEx, KDEyWGF2, lw = 3, label = "WGF")

# savefig(p, "at.pdf")
diagnosticsF(f, KDEx, KDEyWGF1)
diagnosticsF(f, KDEx, KDEyWGF2)
