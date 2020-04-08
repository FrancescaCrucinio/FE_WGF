# push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using KernelDensity;
using Interpolations;
using LinearAlgebra;
# custom packages
using diagnostics;
using smcems;
using wgf;
using samplers;

# set seed
Random.seed!(1234);

# variances and means
mu = [0.5, 0.5];

sigmaF = [0.15 -0.0645; -0.0645 0.43];
sigmaG = [0.45 0.5; 0.5 0.9];
sigmaH = sigmaF + sigmaG;

ifelse(isposdef(sigmaF) & isposdef(sigmaG), "matrices are positive definite",
        "change covariance matrices!")
# data for anaytically tractable example
f(x) = pdf.(MvNormal(mu, sigmaF), x);
h(x) = pdf.(MvNormal(mu, sigmaH), x);
g(x, y) = pdf.(MvNormal(x, sigmaG), y);

# number of iterations
Niter = trunc(Int, 1e04);
# samples from h(y)
M = 10000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 10000;
# regularisation parameter
lambda = 10;

# initial distribution
x0 = 6*rand(2, Nparticles) .- 3;
# run WGF
x, y = wgf_mvnormal(Nparticles, Niter, lambda, x0, M, mu, sigmaH, sigmaG);
p1 = scatter(x[Niter, :], y[Niter, :])


sample = rand(MvNormal(mu, sigmaF), 100000);
p2 = scatter(sample[1, :], sample[2, :])
sampleH = rand(MvNormal(mu, sigmaH), 100000);
p3 = scatter(sampleH[1, :], sampleH[2, :])
plot(p1, p2, p3, layout =(1, 3))


KDEyWGF =  KernelDensity.kde((x[end, :], y[end, :]));
Xbins = range(-0.5, stop = 1.5, length = 1000);
Ybins = range(-1, stop = 2, length = 2000);
res = pdf(KDEyWGF, Ybins, Xbins);
heatmap(Xbins, Ybins, res)
