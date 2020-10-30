push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using wgf;
using samplers;

# set seed
Random.seed!(1234);

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
    trueH = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), KDEx)/3 +
            pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), KDEx)/3;
    # approximated value
    delta = refY[2] - refY[1];
    hatH = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatH[i] = delta*sum(pdf.(Normal.(refY, 0.045), refY[i]).*t);
    end
    kl = kl_divergence(trueH, hatH);
    return kl-a*ent;
end
# true density
trueH = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), KDEx)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), KDEx)/3;

# parameters for WGF
# dt and number of iterations
dt = 1e-03;
Niter = 200;
# number of particles
Nparticles = 500;
# regularisation parameters
alpha = range(0, stop = 1, length = 10);
# initial distribution
x0 = 0.5 .+ randn(1, Nparticles)/10;
# samples from h(y)
M = 500;
muSample = Ysample_gaussian_mixture(100000);

E = zeros(length(alpha), 1);
ise = zeros(length(alpha), 1);
for i=1:length(alpha)
    x = wgf_gaussian_mixture_tamed(Nparticles, dt, Niter, alpha[i], x0, muSample, M, 0.5);
    # KL
    a = alpha[i];
    KDE = phi(x[Niter, :]);
    E[i] = psi(KDE);
    ise[i] = var(trueH .- KDE);
    println("$i")
end
plot(alpha,  E)
plot(alpha, ise)
