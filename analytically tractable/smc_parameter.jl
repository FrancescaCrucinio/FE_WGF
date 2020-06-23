push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
using LaTeXStrings;
# custom packages
using smcems;


function AT_alpha_SMC(target_entropy, interval, threshold)

function remove_non_finite(x)
       return isfinite(x) ? x : zero(x)
end

Niter = 100;
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(-0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
liminf = interval[1];
limsup = interval[2];
x0 = rand(1, Nparticles);

delta_entropy = Inf;
epsilon = (limsup + liminf)/2;
while ((limsup-liminf)>threshold)
    ### WGF
    xSMC, W = smc_AT_approximated_potential(Nparticles, Niter, epsilon, x0, M);
    # kde
    bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
    KDEySMC = weightedKDE(xSMC[end, :], W[end, :], bw, KDEx);
    actual_entropy = -mean(remove_non_finite.(KDEySMC .* log.(KDEySMC)));
    delta_entropy = actual_entropy - target_entropy;
    if (delta_entropy > 0)
        limsup = (limsup + liminf)/2;
    else
        liminf = (limsup + liminf)/2;
    end
    epsilon = (limsup + liminf)/2;
end
return epsilon
end
