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
sigmaF = [0.015 0.0; 0.0 0.043];
sigmaG = [0.45 0.5; 0.5 0.9];
sigmaH = sigmaF + sigmaG;

ifelse(isposdef(sigmaF) & isposdef(sigmaG), "matrices are positive definite",
        "change covariance matrices!")
# data for anaytically tractable example
f(x) = pdf.(MvNormal(mu, sigmaF), x);
h(x) = pdf.(MvNormal(mu, sigmaH), x);
g(x, y) = pdf.(MvNormal(x, sigmaG), y);

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

# initial distribution
x0 = rand(2, Nparticles);
# run WGF
x, y = wgf_mvnormal(N, Niter, lambda, x0, M, mu, sigmaH, sigmaG);
p1 = scatter(x[Niter, :], y[Niter, :])


sample = rand(MvNormal(mu, sigmaF), 100000);
p2 = scatter(sample[1, :], sample[2, :])
plot(p1, p2, layout =(1, 2))