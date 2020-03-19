using StatsPlots;
using Distributions;
using Statistics;
using KernelEstimator;
include("wgf_AT_approximated.jl")

N = 10000;
M = 1000;
dt = 1e-03;
lambda = 1;

x, drift = wgf_AT_approximated(dt, lambda, M, N);
KDEx = range(0, stop = 1, length = 1000);
KDEy = kerneldensity(x[end, :], xeval = KDEx);
p1 = plot(KDEx, KDEy);
p2 = scatter(x[end-1, :], drift[end, :]);
title!("n = end")
plot(p1, p2, layout = (2, 1))

println(mean(x[end, :]))
println(var(x[end, :], corrected=false))
