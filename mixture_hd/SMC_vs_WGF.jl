push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using LinearAlgebra;
# using RCall;
# @rimport ks as rks;
# custom packages
using wgf_prior;
using smcems;
include("sobolev_norm_kde");

# set seed
Random.seed!(1234);

d = 2;
# mixture of Gaussians
means = [0.3*ones(1, d); 0.7*ones(1, d)];
variances = [0.07^2; 0.1^2];
pi = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(d))), (means[2, :], diagm(variances[2]*ones(d)))], [1/3, 2/3]);
sigmaK = 0.15;
mu = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(d) .+ sigmaK^2)), (means[2, :], diagm(variances[2]*ones(d) .+ sigmaK^2))], [1/3, 2/3]);


# parameters for penalised KL
# regularisation parameters
epsilon = 1e-03;
alpha = 0.01;
# number of iterations
Niter = 50;
# time discretisation
dt = 1e-2;
# reference measure
m0 = 0.5;
sigma0 = 0.25;
# number of particles
Nparticles = 10^3;
# sample from μ
muSample = rand(mu, 10^6);
x0 = rand(MvNormal(m0*ones(d), diagm(sigma0*ones(d))), Nparticles);


# SMC-EMS
tSMC = @elapsed begin
xSMC, W, funSMC = smc_mixture_hd(Nparticles, Niter, epsilon, x0, muSample, sigmaK, true);
end

# WGF
dt = 1e-2;
tWGF = @elapsed begin
xWGF, funWGF = wgf_hd_mixture_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, sigmaK, true);
end


sobolev_norm_kde(xSMC, 10, W, epsilon)
sobolev_norm_kde(xWGF, 10, WGFWeights, epsilon)