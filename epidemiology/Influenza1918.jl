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
using DataFrames;
using CSV;
using XLSX;
using Query;
using RCall;
@rimport ks as rks;
# custom packages
using wgf;

# set seed
Random.seed!(1234);
# values at which evaluate KDE
KDEx = range(-2, stop = 2, length = 1000);
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

# DATA
R"""
library(readxl)
df <- read_excel("C:/Users/Francesca/Desktop/WGF/epidemiology/12976_2007_126_MOESM1_ESM.xls", range = "A6:C132")
deaths <- df$Deaths
"""
hHist = Int.(@rget deaths);
hMass = sum(hHist);
hSample = vcat(fill.(1:126, hHist)...)/length(hHist);

Nparticles = 1000;
M = 500;
dt = 1e-3;
Niter = 10000;
x0 = 0.5 .+ 0.2*randn(1, Nparticles);
alpha = 0.01;
x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, hSample, M, 0.5);

# check convergence
KDEyWGF = mapslices(phi, x[2:end, :], dims = 2);
entWGF = mapslices(psi, KDEyWGF, dims = 2);
p1 = plot(entWGF);
# result
# reconstructed incidence
RKDE = rks.kde(x[Niter, :]);
KDEy = abs.(rcopy(RKDE[3]));
KDEx = rcopy(RKDE[2]);
p2 = plot(KDEx, KDEy);
# deaths distribution
RKDEh = rks.kde(hSample);
KDEyh = abs.(rcopy(RKDEh[3]));
KDExh = rcopy(RKDEh[2]);
plot!(p2, KDExh, KDEyh);
p = plot(p1, p2, layout =(2, 1));
p
#
# fSample = walker_sampler(KDEy, hMass);
# cf = counts(fSample);
# R"""
# library(R0)
# data(Germany.1918)
# GT.flu <- generation.time("gamma", c(2.6,1))
# res.R <- estimate.R(Germany.1918, GT=GT.flu, methods= c("EG","ML","SB","TD"))
# res.WGF <- estimate.R($cf, GT=GT.flu, methods= c("EG","ML","SB","TD"))
# res.R
# res.WGF
# """
