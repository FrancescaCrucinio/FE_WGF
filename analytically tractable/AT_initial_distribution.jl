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

# Compare initial distributions

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
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
lambda = 0.025;

# initial distributions
x0 = [0.0*ones(1, Nparticles); 0.5*ones(1, Nparticles);
    1*ones(1, Nparticles); rand(1, Nparticles);
    0.5 .+ sqrt(sigmaF)*randn(1, Nparticles)];

E = zeros(size(x0, 1), Niter);
for i=1:size(x0, 1)
    ### WGF
    x, E[i] =  wgf_AT_E(Nparticles, Niter, lambda, x0[i, :], M, KDEx, refY, f, h, g);
    println("$i finished")
end

# plot
StatsPlots.plot(1:Niter, E, lw = 3)
