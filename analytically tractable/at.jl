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
using wgf;

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
Nparticles = 10000;
# regularisation parameter
lambda = 0.025;


x0 = rand(1, Nparticles);
### WGF
x, _ =  wgf_AT(Nparticles, Niter, lambda, x0, M);
# KDE
KDEyWGF1 =  KernelDensity.kde(x[end, :]);
# evaluate KDE at reference points
KDEyWGFeval1 = pdf(KDEyWGF1, KDEx);
KDEyWGFeval1[KDEyWGFeval1 .< 0] .= 0;

KDEyWGF2 =  KernelDensity.kde(x[end, :], bandwidth=dt);
# evaluate KDE at reference points
KDEyWGFeval2 = pdf(KDEyWGF2, KDEx);
KDEyWGFeval2[KDEyWGFeval2 .< 0] .= 0;
### exact minimiser
variance, _  = AT_exact_minimiser(sigmaG, sigmaH, lambda);
ExactMinimiser(x) = pdf.(Normal(0.5, sqrt(variance)), x);

# plot
p = StatsPlots.plot(f, 0, 1, lw = 3, label = "True f")
StatsPlots.plot!(ExactMinimiser, 0, 1, lw = 3, label = "Exact minimiser")
StatsPlots.plot!(KDEx, KDEyWGFeval1, lw = 3, label = "WGF")
StatsPlots.plot!(KDEx, KDEyWGFeval2, lw = 3, label = "WGF")

# savefig(p, "at.pdf")
diagnosticsF(f, KDEx, KDEyWGFeval1)
diagnosticsF(f, KDEx, KDEyWGFeval2)
