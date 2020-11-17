# push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
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
library(tictoc)
library(fDKDE)
library(readxl)
library(ks)

# get contaminated data
sucrase_Carter1981 <- read_excel("deconvolution/sucrase_Carter1981.xlsx");
W <- sucrase_Carter1981$Pellet;
n <- length(W);

# normal error distribution
errortype="norm";
sigU = sqrt(var(W)/4);
varU=sigU^2;

# DKDE
# Delaigle's estimators
# KDE for mu
h=1.06*sqrt(var(W))*n^(-1/5);
muKDE = kde(W, h = h);
muKDEy = muKDE$estimate;
muKDEx = muKDE$eval.points;

#PI bandwidth of Delaigle and Gijbels
hPI=PI_deconvUknownth4(W,errortype,varU,sigU);
#DKDE estimator
dx = muKDEx[2] - muKDEx[1];
fdec_hPI = fdecUknown(muKDEx,W,hPI,errortype,sigU,dx);
"""

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
    hatMu= zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(pdf.(Normal.(refY, sigU), refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-alpha*ent;
end

# get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
alpha = 0.5;
Nparticles = 500;
dt = 1e-2;
Niter = 100000;
M = 500;
x0 = sample(muSample, Nparticles, replace = true);
tWGF = @elapsed begin
x = wgf_sucrase_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, 0.5, sigU);
end
println("WGF done, $tWGF")

# check convergence
KDEyWGF = mapslices(phi, x, dims = 2);
EWGF = mapslices(psi, KDEyWGF, dims = 2);
plot(EWGF)

# plot
R"""
    # WGF estimator
    KDE_wgf <- kde($x[$Niter, ], eval.points = muKDEx);
    library(ggplot2)
    g <- rep(1:3, , each = length(muKDEx));
    x <- rep(muKDEx, times = 3);
    data <- data.frame(x = x, y = c(muKDEy, fdec_hPI, KDE_wgf$estimate), g = factor(g))
    p <- ggplot(data, aes(x, y, color = g)) +
    geom_line(size = 2) +
    scale_color_manual(values = c("black", "red", "blue"), labels=c(expression(paste("KDE ", mu)), "fdec-hPI", "WGF")) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("sucrase.eps", p, height = 5)
"""
