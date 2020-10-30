push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using Distances;
using DataFrames;
using XLSX;
using RCall;
@rimport ks as rks;
# custom packages
using wgf;

# set seed
Random.seed!(1234);

# sample from μ
R"""
library(R0)
data(Germany.1918)
df <- data.frame(Germany.1918)
deaths <- df$Germany.1918
"""
muCounts = Int.(@rget deaths);
muSample = vcat(fill.(1:length(muCounts), muCounts)...);
# KDE for μ
RKDE = rks.kde(muSample);
muKDEy = abs.(rcopy(RKDE[3]));
# reference values for KL divergence
muKDEx = rcopy(RKDE[2]);

Nparticles = 1000;
M = 1000;
dt = 1e-2;
Niter = 10000;
x0 = sample(muSample, M, replace = true);
alpha = 1;
x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, 0.5);

# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t);
    return abs.(rcopy(RKDE[3]));
end
# function computing entropy
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : zero(x)
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    return ent;
end

# check convergence
KDEyWGF = mapslices(phi, x, dims = 2);
entWGF = mapslices(psi, KDEyWGF, dims = 2);
p1 = plot(entWGF);
# result
# reconstructed incidence
RKDE = rks.kde(x[Niter, :]);
KDEy = abs.(rcopy(RKDE[3]));
KDEx = rcopy(RKDE[2]);
p2 = plot(KDEx, KDEy);
# deaths distribution
plot!(p2, muKDEx, muKDEy);
p = plot(p1, p2, layout =(2, 1));
p

theta = 1/2.6;
kappa = 2.6^2;
reconstruction = x[Niter, :] .+ rand(Gamma(kappa, theta), Nparticles);
