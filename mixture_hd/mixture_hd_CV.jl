push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
# custom packages
using wgf_prior;
using smcems;

# set seed
Random.seed!(1234);

d = 1;
# mixture of Gaussians
means = [0.3 0.7];
variances = [0.07^2; 0.1^2];
pi = MixtureModel(MvNormal, [(means[1]*ones(d), diagm(variances[1]*ones(d))), (means[2]*ones(d), diagm(variances[2]*ones(d)))], [1/3, 2/3]);
sigmaK = 0.15;
mu = MixtureModel(MvNormal, [(means[1]*ones(d), diagm(variances[1]*ones(d) .+ sigmaK^2)), (means[2]*ones(d), diagm(variances[2]*ones(d) .+ sigmaK^2))], [1/3, 2/3]);

# number of iterations
Niter = 50;
# time discretisation
dt = 1e-2;
# reference measure
m0 = 0.5;
sigma0 = 0.25;
# number of particles
Nparticles = 10^4;
# regularisation parameters
alpha = range(0.00001, stop = 0.01, length = 20);
epsilon = range(0.00001, stop = 0.01, length = 20);


L = 10;
EWGF = zeros(length(alpha), L);
ESMC = zeros(length(epsilon), L);
for i=1:length(alpha)
    for l=1:L
        # sample from μ
        muSample = rand(mu, 10^6);
        x0 = rand(mu, Nparticles);
        # SMC-EMS
        xSMC, W, funSMC = smc_mixture_hd(Nparticles, Niter, epsilon[i], x0, muSample, sigmaK, true);
        ESMC[i, l] = funSMC[end];

        # WGF
        xWGF, funWGF = wgf_hd_mixture_tamed(Nparticles, dt, Niter, alpha[i], x0, m0, sigma0, muSample, sigmaK, true);
        EWGF[i, l] = funWGF[end];
        println("$i, $l")
    end
end
plot(alpha,  mean(EWGF, dims = 2), xlabel = "alpha, d = $d")
plot(epsilon,  mean(ESMC, dims = 2), xlabel = "epsilon, d = $d")
alpha[argmin(mean(EWGF, dims = 2))[1]]
epsilon[argmin(mean(ESMC, dims = 2))[1]]
