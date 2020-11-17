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


R"""
library(incidental)
# death counts
death_counts <- spanish_flu$Philadelphia
# fit lognormal to delay distribution
x <- sample(spanish_flu_delay_dist$days, 1000000, replace = TRUE, prob = spanish_flu_delay_dist$proportion)
fit.lognormal <- MASS::fitdistr(x, "log-normal")
ln_meanlog <- fit.lognormal$estimate[1]
ln_sdlog <- fit.lognormal$estimate[2]
"""
# get Gamma parameters
ln_meanlog = @rget ln_meanlog;
ln_sdlog =  @rget ln_sdlog;
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
        hatMu[i] = delta*sum(pdf.(LogNormal(ln_meanlog, ln_sdlog), refY[i] .- KDEx).*t);
    end
    hatMu[iszero.(hatMu)] .= eps();
    kl = kl_divergence(trueMu, hatMu);
    return kl-a*ent;
end

# parameters for WGF
# number of particles
Nparticles = 1000;
# number of samples from μ to draw at each iteration
M = 1000;
# time discretisation
dt = 1e-2;
# number of iterations
Niter = 1000;
# initial distribution
x0 = sample(muSample, M, replace = true) .- 9;
# regularisation parameter
alpha = range(0.00001, stop = 0.01, length = 100);

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
        x = wgf_flu_tamed(Nparticles, dt, Niter, alpha[i], x0, muSample, M, 0.5, ln_meanlog, ln_sdlog);
        # KL
        a = alpha[i];
        KDE = phi(x[Niter, :]);
        E[i, l] = psi(KDE);
        println("$i, $l")
    end
end

plot(alpha, mean(E, dims = 2))
