push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelDensity;
using Random;
using JLD;
# custom packages
using diagnostics;
using wgf;

# load data
d = load("AT_WGF_exact.jld");
varWGF = d["varWGF"];
lambda = d["lambda"];
varExact = d["varExact"];
eWGF = d["eWGF"];
eExact = d["eExact"];


p1 = StatsPlots.plot(lambda, varExact, lw = 3, label = "Exact");
StatsPlots.plot!(p1, lambda, varWGF, lw = 3, label = "WGF");

p2 = StatsPlots.plot(lambda, eExact, lw = 3, label = "Exact");
StatsPlots.plot!(p2, lambda, eWGF, lw = 3, label = "WGF");
