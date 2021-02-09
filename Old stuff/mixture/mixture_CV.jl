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
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using diagnostics;
using wgf;
using samplers;
# set seed
Random.seed!(1234);

# data for gaussian mixture example
rho(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
mu(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
K(x, y) = pdf.(Normal(x, 0.045), y);

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 200;
# number of particles
Nparticles = 500;
# regularisation parameters
alpha = range(0, stop = 0.2, length = 100);

# initial distribution
x0 = 0.5 .+ randn(1, Nparticles)/10;
# samples from μ(y)
M = 500;

# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
a = 1;
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = KDEx);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = mu.(refY);
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(K.(KDEx, refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-a*ent;
end

L = 20;
muSample = Ysample_gaussian_mixture(100000);
muSample = reshape(muSample, (L, Int(length(muSample)/L)));


E = zeros(length(alpha), L);
for i=1:length(alpha)
    for l=1:L
        # get reduced sample
        muSampleL = muSample[1:end .!= l, :];
        muSampleL = muSampleL[:];
        # WGF
        x = wgf_gaussian_mixture_tamed(Nparticles, dt, Niter, alpha[i], x0, muSampleL, M, 0.5);
        # KL
        a = alpha[i];
        KDE = phi(x[Niter, :]);
        E[i, l] = psi(KDE);
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
