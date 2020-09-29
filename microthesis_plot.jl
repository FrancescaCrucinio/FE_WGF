push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
using Distances;
using LaTeXStrings;
using RCall;
@rimport ks as rks
# custom packages
using diagnostics;
using wgf;

# set seed
Random.seed!(1234);

# data for gaussian mixture example
f(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
h(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
g(x, y) = pdf.(Normal(x, 0.045), y);

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
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
alpha = 0.0001;
x0 = 0.5*ones(1, Nparticles);
# run WGF
x =  wgf_gaussian_mixture(Nparticles, dt, Niter, alpha, x0, M);
# KDE
KDEyWGF = [zeros(1, Nparticles); mapslices(phi, x[2:Niter, :], dims = 2)];
p = StatsPlots.plot(f, xlims = (0,1), lw = 10, label = "f", color=:black, legend=false, grid=false)
scatter!(x[Niter, :], zeros(Nparticles, 1), markersize = 10, markercolor = :blue)
# vline!([0.5], color=:blue, lw=3)
StatsPlots.plot!(KDEx, KDEyWGF[Niter, :], lw = 8, label = "WGF", color=:blue, line=:dashdot)
savefig(p, "mixture100.pdf")
