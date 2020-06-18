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
lambda = [0.01 0.025 0.05];

x0 = 0.5*ones(1, Nparticles);
# x0 = rand(1, Nparticles);
KDEyWGF1 =zeros(1000, length(lambda));
Threads.@threads for i=1:length(lambda)
    ### WGF
    x, _ =  wgf_AT(Nparticles, dt, Niter, lambda[i], x0, M);
    # KDE
    # optimal bandwidth Gaussian
    KDEyWGF1[:, i] =  KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=bwnormal(x[end,:]));
    # bw = dt
    # KDEyWGF2 =  KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=dt);
end

# plot
pyplot()
labels = [L"$\alpha=0.01$" L"$\alpha=0.025$" L"$\alpha=0.05$"];
p = StatsPlots.plot(f, 0, 1, lw = 3, label = L"True $\rho$", color=:black,
    legendfontsize = 10);
StatsPlots.plot!(KDEx, KDEyWGF1, lw = 3, label = labels, color=[1 2 3])

savefig(p, "at_alpha.pdf")
