push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
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

# set seed
set.seed(1234);

# get data
sucrase_Carter1981 <- read_excel("deconvolution/sucrase_Carter1981.xlsx");
X <- sucrase_Carter1981$Pellet;
n <- length(X);
#Noise to signal ratio=varU/varX
NSR=1/3

#Specify error distribution (normal or Laplace in this case) and generate data from this error distribution
errortype="norm";
#normal case
if (errortype=="norm")
{sigU=sqrt(NSR*var(X));
U=rnorm(n,0,sigU);
varU=sigU^2;}

#Laplace case
if (errortype=="Lap")
{sigU=sqrt(NSR*var(X)/2);
varU=2*sigU^2;
U=rlap(sigU,1,n);}

#Contaminated data
W=as.vector(X+U);

# DKDE
# Delaigle's estimators
# KDE for μ
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
    trueH = @rget muKDEy;
    refY = @rget muKDEx;
    # approximated value
    delta = refY[2] - refY[1];
    hatH = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatH[i] = delta*sum(pdf.(Laplace.(refY, sigU), refY[i]).*t);
    end
    kl = kl_divergence(trueH, hatH);
    return kl-alpha*ent;
end

get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
a = 0.5;
alpha = 0.07;
Nparticles = 1000;
dt = 1e-2;
Niter = 10000;
M = 1000;
x0 = sample(muSample, Nparticles, replace = true);
tWGF = @elapsed begin
x = wgf_sucrase_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, a, sigU);
end
println("WGF done, $tWGF")

# check convergence
KDEyWGF = mapslices(phi, x, dims = 2);
EWGF = mapslices(psi, KDEyWGF, dims = 2);
plot(EWGF)

# # plot
# R"""
#     library(ggplot2)
#     data <- data.frame(x = 1:$Niter, y = $entWGF)
#     p1 <- ggplot(data, aes(x, y)) +
#     geom_line(size = 2) +
#     theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
#     # ggsave("sucrase_convergence.eps", p1,  height=5)
#     g <- rep(1:2, , each= c(length($muKDEx), length($KDEx)));
#     glabels <- c(expression(mu(y)), expression(rho(x)));
#     data <- data.frame(x = c($muKDEx, $KDEx), y = c($muKDEy, $KDEy), g = g)
#     p2 <- ggplot(data, aes(x, y, color = factor(g))) +
#     geom_line(size = 2) +
#     scale_colour_manual(values = c("red", "blue"), labels=glabels) +
#     theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
#     # ggsave("sucrase.eps", p2,  height=5)
# """
