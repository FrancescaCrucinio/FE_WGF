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
using wgf;


function AT_alpha_WGF(target_entropy, interval, threshold, dt)

function remove_non_finite(x)
       return isfinite(x) ? x : zero(x)
end

Niter = 100;
# samples from h(y)
M = 500;
# values at which evaluate KDE
KDEx = range(-0, stop = 1, length = 1000);
# number of particles
Nparticles = 500;
liminf = interval[1];
limsup = interval[2];
x0 = 0.5*ones(1, Nparticles);

delta_entropy = Inf;
lambda = (limsup + liminf)/2;
while ((limsup-liminf)>threshold)
    ### WGF
    xWGF, _ =  wgf_AT(Nparticles, dt, Niter, lambda, x0, M);
    # KDE
    # optimal bandwidth Gaussian
    KDEyWGF =  KernelEstimator.kerneldensity(xWGF[end,:], xeval=KDEx, h=bwnormal(xWGF[end,:]));
    actual_entropy = -mean(remove_non_finite.(KDEyWGF .* log.(KDEyWGF)));
    delta_entropy = actual_entropy - target_entropy;
    if (delta_entropy > 0)
        limsup = (limsup + liminf)/2;
    else
        liminf = (limsup + liminf)/2;
    end
    lambda = (limsup + liminf)/2;
end
return lambda
end
