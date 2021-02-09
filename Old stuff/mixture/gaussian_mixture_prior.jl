push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using StatsPlots;
using RCall;
@rimport ks as rks
# custom packages
using diagnostics;
using wgf;
using wgf_prior;
using samplers;

# set seed
Random.seed!(1234);

# data for gaussian mixture example
rho(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
mu(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
K(x, y) = pdf.(Normal(x, 0.045), y);

# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);

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
# sample from h(y)
muSample = Ysample_gaussian_mixture(100000);
# number of particles
Nparticles = 1000;
# regularisation parameter
alpha = 0.01;
x0 = 0.5 .+ randn(1, Nparticles)/10;
# run WGF
x =  wgf_gaussian_mixture_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, 0.5);
KDEyWGF = phi(x[Niter, :]);
x_prior =  wgf_prior_gaussian_mixture_tamed(Nparticles, dt, Niter, alpha, x0, 1/10, 0.5, muSample, M, 0.5);
x_pda, time =  pda_gaussian_mixture_tamed(Nparticles, dt/10, 3*Niter, alpha, x0, 1/10, 0.5, muSample, M, 0.5);
KDEyWGF_prior = phi(x_prior[Niter, :]);
KDEyWGF_pda = phi(x_pda[Niter, :]);
plot(rho, 0, 1)
plot!(KDEx, KDEyWGF)
plot!(KDEx, KDEyWGF_prior)
plot!(KDEx, KDEyWGF_pda)
