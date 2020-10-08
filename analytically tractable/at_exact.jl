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
using LaTeXStrings;
using RCall;
@rimport ks as rks
# custom packages
using diagnostics;
using wgf;

# Plot AT example and exact minimiser
# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
h(x) = pdf.(Normal(0.5, sqrt(sigmaH)), x);
g(x, y) = pdf.(Normal(x, sqrt(sigmaG)), y);

# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = KDEx);
    return abs.(rcopy(RKDE[3]));
end

# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(-0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
alpha = 0.05;

x0 = 0.5 .+ randn(1, Nparticles)/10;
# x0 = rand(1, Nparticles);
### WGF
x, _ =  wgf_AT_tamed(Nparticles, dt, Niter, alpha, x0, M, 0.5);
# KDE
# optimal bandwidth Gaussian
KDEyWGF =  KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=bwnormal(x[end,:]));

# exact minimiser
variance, _  = AT_exact_minimiser(sigmaG, sigmaH, alpha);
ExactMinimiser = pdf.(Normal(0.5, sqrt(variance)), KDEx);

# solution
solution = f.(KDEx);
R"""
    library(ggplot2)
    glabels <- c(expression(rho), expression(rho[alpha]), expression(rho[alpha]^N));
    x = rep($KDEx, 3);
    g <- rep(1:3, , each= length($KDEx));
    data <- data.frame(x = x, y = c($solution, $ExactMinimiser, $KDEyWGF), g = g)
    p <- ggplot(data, aes(x, y, group = factor(g), color = factor(g))) +
    geom_line(size = 2) +
    scale_colour_manual(values = c("black", "red", "blue"), labels=glabels) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    ggsave("at_exact_min.eps", p, height=5)
"""
