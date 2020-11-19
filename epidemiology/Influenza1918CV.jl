#push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
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
using wgf;

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

a = 1;
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = KDEx);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = muKDEy;
    refY = KDEx;
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(K.(KDEx, refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-a*ent;
end

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-1;
# number of iterations
Niter = 5000;
# initial distribution
x0 = sample(muSample, M, replace = true) .- 9;
# regularisation parameter
alpha = range(0.001, stop = 0.07, length = 10);

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
        # WGF
        x = wgf_flu_tamed(Nparticles, dt, Niter, alpha[i], x0, muSample, M, 0.5);
        # KL
        a = alpha[i];
        KDE = phi(x[Niter, :]);
        E[i, l] = psi(KDE);
        println("$i, $l")
    end
end

plot(alpha, mean(E, dims = 2))
