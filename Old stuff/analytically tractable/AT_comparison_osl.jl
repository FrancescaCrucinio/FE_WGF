push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
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

include("osl_em.jl")
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

# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(-0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
alpha = 0.01;

x0 = rand(1, Nparticles);
# x0 = rand(1, Nparticles);
### WGF
x, _ =  wgf_AT(Nparticles, dt, Niter, alpha, x0, M);
# KDE
# optimal bandwidth Gaussian
KDEyWGF =  KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=bwnormal(x[end,:]));

### exact minimiser
variance, _  = AT_exact_minimiser(sigmaG, sigmaH, alpha);
ExactMinimiser(x) = pdf.(Normal(0.5, sqrt(variance)), x);


Nbins = 100;
x0 = rand(1, Nbins);
# bin centers
pp = range(1/(2*Nbins), stop = 1, step = 1/Nbins);
# discretisation of h for EM, EMS and IB
hDisc = h(pp);
# discretisation of g for EM, EMS and IB
gDisc = zeros(Nbins, Nbins);
for i=1:Nbins
   for j=1:Nbins
      gDisc[i, j] = g(pp[j], pp[i]);
   end
end

fEM = osl_em(gDisc, hDisc, Niter, x0, alpha);
# plot
pyplot()
p = StatsPlots.plot(f, 0, 1, lw = 5, label = L"True $\rho$", color=:black,
    legendfontsize = 10);
StatsPlots.plot!(ExactMinimiser, 0, 1, lw = 3, label = "Exact minimiser", color=1);
StatsPlots.plot!(KDEx, KDEyWGF, lw = 3, label = "WGF", color=2);
StatsPlots.plot!(pp, fEM[end, :], lw = 3, label = "OSL-EM", color=3);

diagnosticsF(f, KDEx, KDEyWGF)
diagnosticsF(f, pp, fEM[end, :])
