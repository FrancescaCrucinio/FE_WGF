push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
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
mu = [0, 0];

sigmaF = [0.2 0; 0 0.2];
sigmaG = [1 0.5; 0.5 1];
sigmaH = sigmaF + sigmaG;

ifelse(isposdef(sigmaF) & isposdef(sigmaG), "matrices are positive definite",
        "change covariance matrices!")
# data for anaytically tractable example
f(x) = pdf(MvNormal(mu, sigmaF), x);
h(x) = pdf.(MvNormal(mu, sigmaH), x);
g(x, y) = pdf.(MvNormal(x, sigmaG), y);

# dt and number of iterations
dt = 1e-02;
Niter = 1500;
# samples from h(y)
M = 5000;
# number of particles
Nparticles = 5000;
# regularisation parameter
lambda = 0.005;

# initial distribution
x0 = 2*rand(2, Nparticles) .- 1;
# run WGF
x, y = wgf_mvnormal(Nparticles, dt, Niter, lambda, x0, M, mu, sigmaH, sigmaG);
p3 = scatter(x[end, :], y[end, :]);
# KDE
KDEyWGF =  KernelDensity.kde((x[end, :], y[end, :]));
Xbins = range(-1, stop = 1, length = 1000);
Ybins = range(-1, stop = 1, length = 1000);
res = pdf(KDEyWGF, Ybins, Xbins);
p1 = heatmap(Xbins, Ybins, res);

fplot = zeros(1000, 1000);
for i=1:1000
    for j=1:1000
        fplot[j, i] = f([Xbins[i]; Ybins[j]]);
    end
end
p2 = heatmap(Xbins, Ybins, fplot);
p = plot(p1, p2, layout=(2, 1));
savefig(p, "mvn.pdf");