push!(LOAD_PATH, "/home/u1693998/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using Distances;
using KernelEstimator;
# custom packages
using wgf;

# set seed
Random.seed!(1234);

# reference values for KDE
muKDEx = range(-189, stop = 540, length = 400);
# KDE for μ
bw = 1.06*sqrt(var(muSample))*length(muSample)^(-1/5);
muKDEy = kerneldensity(muSample, xeval = muKDEx, h = bw);
# sample from μ
muSample =  [70.00, 55.43, 18.87, 40.41, 57.43, 31.14, 70.10, 137.56, 221.20,
    276.43, 316.00, 75.56, 277.30, 331.50, 133.74, 221.50, 132.93, 85.38,
    142.34, 294.63, 262.52, 183.56, 86.12, 226.55];
# normal error distribution
sigU = sqrt(var(muSample)/4);

# function computing KDE
function phi(t)
    bw = 1.06*sqrt(var(t))*length(t)^(-1/5);
    KDE = kerneldensity(t, xeval=muKDEx, h = bw);
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = muKDEy;
    refY = muKDEx;
    # approximated value
    delta = refY[2] - refY[1];
    hatMu= zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(pdf.(Normal.(refY, sigU), refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-alpha*ent;
end

# parameters for WGF
alpha = 0.5;
Nparticles = 1000;
dt = 1e-2;
Niter = 10;
M = 1000;
x0 = sample(muSample, Nparticles, replace = true);
tWGF = @elapsed begin
x = wgf_sucrase_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, 0.5, sigU);
end
println("WGF done, $tWGF")

# check convergence
KDEyWGF = mapslices(phi, x, dims = 2);
EWGF = mapslices(psi, KDEyWGF, dims = 2);
plot(EWGF)
