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
@rimport ks as rks
# custom packages
using wgf;
using samplers;

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
daily_cases = [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	1,	0,	0,	0,	1,	0,	0,	0,	1,	1,	0,	1,	2,	0,	0,	1,	2,	0,	2,	3,	6,	7,	6,	5,	11,	11,	11,	6,	9,	19,	8,	9,	4,	11,	10,	9,	11,	15,	14,	13,	14,	18,	18,	16,	23,	22,	30,	25,	13,	30,	25,	31,	17,	46,	34,	24,	27,	37,	29,	44,	26,	39,	42,	27,	28,	34,	41,	49,	45,	41,	53,	58,	40,	53,	47,	32,	25,	39,	42,	38,	48,	48,	50];
hSample = vcat(fill.(1:130, daily_cases)...)/130;

Nparticles = 5000;
M = 5000;
dt = 1e-3;
Niter = 500;
x0 = sample(hSample, Nparticles, replace = true);
alpha = 0.01;
x = wgf_HIV_tamed(Nparticles, dt, Niter, alpha, x0, hSample, M, 0.5);


KDEyWGF = mapslices(phi, x[2:end, :], dims = 2);
entWGF = mapslices(psi, KDEyWGF, dims = 2);

plot(entWGF)
# last time step
RKDE = rks.kde(x[Niter, :]);
KDEx = rcopy(RKDE[2]);
KDEy = abs.(rcopy(RKDE[3]));
plot(KDEx, KDEy)
#
# refY = range(0, stop = 1, length = 1000);
# delta = refY[2] - refY[1];
# hatH = zeros(1, length(refY));
# # convolution with approximated f
# # this gives the approximated value
# kappa = 2.516;
# lambda = 8/ log(2)^(1/kappa);
# for i=1:length(refY)
#     hatH[i] = delta*sum(pdf.(Weibull(kappa, lambda), refY[i] .- KDEx .+ 1).*KDEy);
# end
