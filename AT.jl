using StatsPlots;
using Distributions;
using Statistics;
using KernelEstimator;
include("wgf_AT_approximated.jl")

sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
N = 1000;
M = 1000;
dt = 1e-03;
lambda = 0.9;

sigmalambda = abs(exact_minimizer(sigmaG, sigmaH, lambda));
flambda(x) =  pdf.(Normal(0.5, sqrt(sigmalambda)), x);
x, drift = wgf_AT_approximated(dt, lambda, M, N);
KDEx = range(0, stop = 1, length = 1000);
KDEy = kerneldensity(x[end, :], xeval = KDEx);

println(mean(x[end, :]))
println(var(x[end, :], corrected=false))
println(0.043^2)

plot(f, 0, 1)
plot!(KDEx, KDEy)
plot!(flambda, 0, 1)
xlabel!("x")
ylabel!("f(x)")
title!("Reconstruction via WGF, N=$N, lambda=$lambda, dt=$dt")
