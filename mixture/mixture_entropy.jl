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
using RCall;
@rimport ks as rks
# custom packages
using diagnostics;
using smcems;
using wgf;
using samplers;

# set seed
Random.seed!(1234);

# data for anaytically tractable example
# data for gaussian mixture example
f(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
h(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
g(x, y) = pdf.(Normal(x, 0.045), y);

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 1000;
# number of particles
Nparticles = 1000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# regularisation parameters
epsilon = 1e-5;
alpha = 1e-2;

# initial distributions
x0SMC = rand(1, Nparticles);
x0WGF = 0.5*ones(1, Nparticles);
# sample from h(y)
hSample = Ysample_gaussian_mixture(M);
# SMC
xSMC, W = smc_gaussian_mixture(Nparticles, Niter, epsilon, x0SMC, hSample, M);
# kde
bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
RKDESMC = rks.kde(x = xSMC[end,:], var"h" = bw, var"eval.points" = KDEx, var"w" = W[end, :]);
KDEySMC =  abs.(rcopy(RKDESMC[3]));
# WGF
xWGF = wgf_gaussian_mixture(Nparticles, dt, Niter, alpha, x0WGF, hSample, M);
RKDEWGF = rks.kde(x = xWGF[end,:], var"eval.points" = KDEx);
KDEyWGF =  abs.(rcopy(RKDEWGF[3]));

# solution
solution = f.(KDEx);

# plot
R"""
    library(ggplot2)
    x = rep($KDEx, 3);
    g <- rep(1:3, , each= length($KDEx));
    glabels <- c(expression(rho(x)), "SMC-EMS", "WGF");
    data <- data.frame(x = x, y = c($solution, $KDEySMC, $KDEyWGF), g = g)
    p <- ggplot(data, aes(x, y, group = factor(g), color = factor(g))) +
    geom_line(size = 2) +
    scale_colour_manual(values = c("black", "blue", "red"), labels=glabels) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("mixture_entropy.eps", p,  height=5)
"""
diagnosticsF(f, KDEx, solution)
diagnosticsF(f, KDEx, KDEySMC)
diagnosticsF(f, KDEx, KDEyWGF)
