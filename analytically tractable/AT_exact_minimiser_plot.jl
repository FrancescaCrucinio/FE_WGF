push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using RCall;
# custom packages
using wgf;

# Plot exact minimiser for AT example
# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;

alpha = range(0, stop = 0.99, length = 1000);

sigma, E = AT_exact_minimiser(sigmaG, sigmaH, α);

R"""
    library(ggplot2)
    data <- data.frame(alpha = $alpha, variance = $sigma, E = $E)
    p1 <- ggplot(data, aes(alpha, variance)) +
    geom_line(size = 2) +
    theme(axis.title=element_blank(), text = element_text(size=20))
    p2 <-  ggplot(data, aes(alpha, E)) +
    geom_line(size = 2) +
    theme(axis.title=element_blank(), , text = element_text(size=20))
    ggsave("at_exact_variance.eps", p1)
    ggsave("at_exact_E.eps", p2)
"""
