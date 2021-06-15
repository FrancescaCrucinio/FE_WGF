push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using RCall;
@rimport ks as rks
# custom packages
include("wgf_AT_tamed.jl")

# set seed
Random.seed!(1234);

# Gaussian toy model

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 100;
# number of particles
Nparticles = 100;
# regularisation parameters
alpha = range(0, stop = 0.001, length = 100);
# prior mean = mean of μ
m0 = 0;
sigma0 = 0.1;

L = 10;
E = zeros(length(alpha), L);
for i=1:length(alpha)
    for l=1:L
        muSample = Ysample_gaussian_mixture(10^3);
        x0 = sample(muSample, Nparticles, replace = !(Nparticles <= 10^3));
        # size of sample from μ
        M = min(Nparticles, length(muSample));
        # WGF
        x, fun = wgf_AT_tamed(Nparticles, dt, Niter, alpha[i], x0, m0, sigma0, muSample, M, "normal");
        # estimate functional
        E[i, l] = fun[end];
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
