#push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using StatsPlots;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using wgf;

# set seed
Random.seed!(1234);

R"""
library(tictoc)
library(readxl)
library(ks)

# get contaminated data
sucrase_Carter1981 <- read_excel("deconvolution/sucrase_Carter1981.xlsx");
W <- sucrase_Carter1981$Pellet;
n <- length(W);

# Laplace error distribution
errortype="Lap";
varU = var(W)/4;
sigU = sqrt(varU/2);

# DKDE
# Delaigle's estimators
# KDE for mu
h=1.06*sqrt(var(W))*n^(-1/5);
muKDE = kde(W, h = h);
muKDEy = muKDE$estimate;
muKDEx = muKDE$eval.points;
bw = muKDE$h;
"""

a = 1;
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = @rget muKDEx);
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
    trueMu = @rget muKDEy;
    refY = @rget muKDEx;
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(pdf.(Laplace.(refY, sigU), refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-a*ent;
end

# get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
alpha = range(0.16, stop = 0.25, length = 5);
Nparticles = 200;
dt = 1e-1;
Niter = 50000;
M = 200;
R"""
# use KDE to sample initial distribution
means <- sample(W, $Nparticles, replace = TRUE)
initial_dist <- rnorm($Nparticles, mean = means, sd = bw)
"""
x0 = @rget initial_dist;
# divide muSample into groups
L = 24;
muSample = reshape(muSample, (L, Int(length(muSample)/L)));


E = zeros(length(alpha), L);
Threads.@threads for i=1:length(alpha)
    @simd for l=1:L
        # get reduced sample
        muSampleL = muSample[1:end .!= l, :];
        # WGF
        x = wgf_sucrase_tamed(Nparticles, dt, Niter, alpha[i], x0, muSample, M, 0.5, sigU);
        # KL
        a = alpha[i];
        KDE = phi(x[Niter, :]);
        E[i, l] = psi(KDE);
        println("$i, $l")
    end
end
save("sucraseCV_03Nov2020.jld", "alpha", alpha, "E", E, "dt", dt,
    "Nparticles", Nparticles, "Niter", Niter);
plot(alpha,  mean(E, dims = 2))
