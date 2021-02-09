push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using RCall;
@rimport ks as rks
# custom packages
using wgf;

# Plot AT example and exact minimiser
# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaK = 0.045^2;
sigmaRho = 0.043^2;
sigmaMu = sigmaRho + sigmaK;
rho(x) = pdf.(Normal(0.5, sqrt(sigmaRho)), x);
mu(x) = pdf.(Normal(0.5, sqrt(sigmaMu)), x);
K(x, y) = pdf.(Normal(x, sqrt(sigmaK)), y);
# regularisation parameter
alpha = range(0, stop = 0.99, length = 100);

variance, E  = AT_exact_minimiser(sigmaK, sigmaMu, alpha);
# plot
R"""
    library(ggplot2)
    data <- data.frame(x = $alpha, y = $E, t = $variance)
    p1 <- ggplot(data, aes(x, y)) +
    geom_line(size = 1) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("AT_sensitivity_E.eps", p1,  height=5)
    p3 <- ggplot(data, aes(x, t)) +
    geom_line(size = 1) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("AT_sensitivity_var.eps", p3,  height=5)
"""
