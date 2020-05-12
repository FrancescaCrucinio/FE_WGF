push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
# custom packages
using wgf;

# Plot exact minimiser for AT example

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;

α = range(0, stop = 0.95, length = 1000);

sigma, E = AT_exact_minimiser(sigmaG, sigmaH, α);

p1 = plot(α, sigma, lw = 3);
savefig(p1, "at_exact_variance.pdf")
p2 = plot(α, E, lw = 3);
savefig(p2, "at_exact_E.pdf")
