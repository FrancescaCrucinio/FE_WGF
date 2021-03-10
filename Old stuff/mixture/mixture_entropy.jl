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
using diagnostics;
using smcems;
using wgf;
using samplers;

# set seed
Random.seed!(1234);

# data for anaytically tractable example
# data for gaussian mixture example
rho(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
mu(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
K(x, y) = pdf.(Normal(x, 0.045), y);

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 500;
# number of particles
Nparticles = 500;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 100);
# regularisation parameters
epsilon = 1e-3;
alpha = 1e-2;

# initial distributions
x0SMC = rand(1, Nparticles);
x0WGF = 0.5*ones(1, Nparticles);
# samples from h(y)
muSample = Ysample_gaussian_mixture(100000);
# SMC
# N = 1000
xSMC1, W1 = smc_gaussian_mixture(Nparticles, Niter, epsilon, x0SMC, muSample, Nparticles);
# kde
bw1 = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC1[Niter, :], W1[Niter, :])^2);
RKDESMC1 = rks.kde(x = xSMC1[end,:], var"h" = bw1, var"eval.points" = KDEx, var"w" = W1[end, :]);
KDEySMC1 =  abs.(rcopy(RKDESMC1[3]));
# N = 10000
x0SMC = rand(1, Nparticles*2);
xSMC2, W2 = smc_gaussian_mixture(Nparticles*2, Niter, epsilon, x0SMC, muSample, Nparticles*2);
# kde
bw2 = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC2[Niter, :], W2[Niter, :])^2);
RKDESMC2 = rks.kde(x = xSMC2[end,:], var"h" = bw2, var"eval.points" = KDEx, var"w" = W2[end, :]);
KDEySMC2 =  abs.(rcopy(RKDESMC2[3]));
# WGF
xWGF = wgf_gaussian_mixture_tamed(Nparticles, dt, Niter, alpha, x0WGF, muSample, Nparticles, 0.5);
RKDEWGF = rks.kde(x = xWGF[end,:], var"eval.points" = KDEx);
KDEyWGF =  abs.(rcopy(RKDEWGF[3]));

# solution
solution = rho.(KDEx);

# plot
R"""
    library(ggplot2)
    x = rep($KDEx, 4);
    g <- rep(1:4, , each= length($KDEx));
    glabels <- c(expression(rho(x)), "SMCEMS, N=500", "SMCEMS, N=1000", "WGF, N=500");
    data <- data.frame(x = x, y = c($solution, $KDEySMC1, $KDEySMC2, $KDEyWGF), g = g)
    p <- ggplot(data, aes(x, y, group = factor(g), color = factor(g))) +
    geom_line(size = 2) +
    scale_colour_manual(values = c("black", "blue", "dodgerblue", "red"), labels=glabels) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("mixture_N.eps", p,  height=5)
"""

var(solution .- KDEySMC1)
var(solution .- KDEySMC2)
var(solution .- KDEyWGF)
