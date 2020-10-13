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
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using diagnostics;
using smcems;
using wgf;
using samplers;
include("mixture/mixture_CValpha.jl")
# set seed
Random.seed!(1234);

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 200;
# number of particles
Nparticles = 500;
# regularisation parameters
alpha = range(0, stop = 0.2, length = 100);

# initial distribution
x0 = 0.5 .+ randn(1, Nparticles)/10;
# samples from h(y)
M = 500;
hSample = Ysample_gaussian_mixture(100000);
hSample = reshape(hSample, (20, 5000));

E = zeros(1, length(alpha));
for i=1:length(alpha)
    E[i] = CValpha(Nparticles, dt, Niter, alpha[i], x0, hSample, M);
end
plot(alpha, E[:])
