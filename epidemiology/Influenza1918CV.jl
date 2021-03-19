push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using Distances;
using RCall;
@rimport ks as rks;
# custom packages
using wgf_prior;

# set seed
Random.seed!(1234);

# fitted Gaussian approximating K
K(x, y) = 0.595*pdf.(Normal(8.63, 2.56), y .- x) +
        0.405*pdf.(Normal(15.24, 5.39), y .- x);

R"""
library(incidental)
# death counts
death_counts <- spanish_flu$Philadelphia
"""
# get counts from μ
muCounts = Int.(@rget death_counts);
# get sample from μ
muSample = vcat(fill.(1:length(muCounts), muCounts)...);
# shuffle sample
shuffle!(muSample);
# x axis = time (122 days)
KDEx = 1:length(muCounts);
# KDE for μ
RKDE = rks.kde(muSample, var"eval.points" = KDEx);
muKDEy = abs.(rcopy(RKDE[3]));

# functional approximation
function psi(piSample, a, m0, sigma0)
    loglik = zeros(1, length(muSample));
    for i=1:length(muSample)
        loglik[i] = mean(K.(piSample, muSample[i]));
    end
    loglik = -log.(loglik);
    kl = mean(loglik);
    prior = pdf.(Normal(m0, sigma0), piSample);
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    kl_prior = mean(log.(pihat./prior));
    return kl+a*kl_prior;
end

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-1;
# number of iterations
Niter = 3000;
# regularisation parameters
alpha = range(0.0001, stop = 0.001, length = 10);

# divide muSample into groups
L = 5;
# add one element at random to allow division
muSample = [muSample; sample(muSample, 1)];
muSample = reshape(muSample, (L, Int64(length(muSample)/L)));


E = zeros(length(alpha), L);
Threads.@threads for i=1:length(alpha)
    @simd for l=1:L
        # get reduced sample
        muSampleL = muSample[1:end .!= l, :];
        muSampleL = muSampleL[:];
        # prior mean = mean of μ shifted back by 9 days
        # initial distribution
        x0 = sample(muSampleL, Nparticles, replace = false) .- 9;
        m0 = mean(muSampleL) - 9;
        sigma0 = std(muSampleL);
        # WGF
        x = wgf_flu_tamed(Nparticles, dt, Niter, alpha[i], x0, m0, sigma0, muSample, M);
        # functional
        E[i, l] = psi(x[Niter, :], alpha[i], m0, sigma0);
        println("$i, $l")
    end
end

plot(alpha, mean(E, dims = 2))
