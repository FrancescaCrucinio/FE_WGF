push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelDensity;
using Random;
using JLD;
# custom packages
using diagnostics;
using wgfserver;

# Plot AT example and exact ExactMinimiser

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
Niter = trunc(Int, 1/dt);
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
lambda = 0.025;


x0 = rand(1, Nparticles);
### WGF
x, _ =  wgf_AT(Nparticles, Niter, lambda, x0, M);
# KDE
KDEyWGF =  KernelDensity.kde(x[end, :]);
# evaluate KDE at reference points
KDEyWGFeval = pdf(KDEyWGF, KDEx);

### exact minimiser
variance, _  = AT_exact_minimiser(sigmaG, sigmaH, lambda);
ExactMinimiser(x) = pdf.(Normal(0.5, sqrt(variance)), x);

# plot
p = StatsPlots.plot(f, 0, 1, lw = 3, label = "True f")
StatsPlots.plot!(ExactMinimiser, 0, 1, lw = 3, label = "Exact minimiser")
StatsPlots.plot!(KDEx, KDEyWGFeval, lw = 3, label = "WGF")

# savefig(p, "at.pdf")
