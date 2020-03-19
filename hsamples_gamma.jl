using StatsPlots;
using Distributions;
using Statistics;
using KernelEstimator;
include("wgf_gamma.jl")

N = 10000;
M = 1000;
dt = 1e-04;
lambda = 0.5;

x, drift = wgf_gamma(dt, lambda, M, N);
KDEx = range(-0.5, stop = 5, length = 2000);
KDEy = kerneldensity(x[end, :], xeval = KDEx);
p1 = plot(KDEx, KDEy);
p2 = scatter(x[end-1, :], drift[end, :]);
title!("n = end")
plot(p1, p2, layout = (2, 1))

println(mean(x[end, :]))
println(var(x[end, :], corrected=false))
