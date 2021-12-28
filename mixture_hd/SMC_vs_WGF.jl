push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "/Users/francescacrucinio/Documents/WGF/myModules")

# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using DelimitedFiles;
# custom packages
using wgf_prior;
using smcems;
include("mixture_hd_stats.jl")


# set seed
Random.seed!(1234);
# dimension
d = 5;
# mixture of Gaussians
means = [0.3 0.7];
variances = [0.07^2; 0.1^2];
# pi = MixtureModel(MvNormal, [(means[1]*ones(d), diagm(variances[1]*ones(d))), (means[2]*ones(d), diagm(variances[2]*ones(d)))], [1/3, 2/3]);
sigmaK = 0.15;
mu = MixtureModel(MvNormal, [(means[1]*ones(d), diagm(variances[1]*ones(d) .+ sigmaK^2)), (means[2]*ones(d), diagm(variances[2]*ones(d) .+ sigmaK^2))], [1/3, 2/3]);
# true mean, variance and probability of quadrant
m = means[1]/3 + 2*means[2]/3;
v = variances[1]/3 + 2*variances[2]/3 + means[1]^2/3 + 2*means[2]^2/3 - m^2;
p = ((cdf(Normal(means[1], variances[1]), 0.5) - cdf(Normal(means[1], variances[1]), 0) +
    2*(cdf(Normal(means[2], variances[2]), 0.5) - cdf(Normal(means[2], variances[2]), 0)))/3)^d;
# number of iterations
Niter = 50;
# time discretisation
dt = 1e-2;
# reference measure
m0 = 0.5;
sigma0 = 0.25;
# number of particles
Nparticles = 10^3;
# regularisation parameters
epsilon = [8e-03 1e-05 2e-3 5e-3 5e-3 5e-3 7e-03 6e-03 7e-03 1e-02];
alpha = [2.5e-3 5e-4 1e-5 1e-5 5e-5 5e-4 5e-4 1e-05 1e-05 1e-05];
# number of replicates
Nrep = 100;
tSMC = zeros(Nrep);
statsSMC = zeros(Nrep, 5);
# entSMC = zeros(Nrep);
tWGF = zeros(Nrep);
statsWGF = zeros(Nrep, 5);
# entWGF = zeros(Nrep);
for j=1:Nrep
    # sample from μ
    muSample = rand(mu, 10^6);
    # initial distribution
    x0 = rand(mu, Nparticles);
    # SMC-EMS
    tSMC[j] = @elapsed begin
    xSMC, W, _ = smc_mixture_hd(Nparticles, Niter, epsilon[d], x0, muSample, sigmaK, false);
    end
    # entSMC[j] = mean(log.(mixture_hd_kde_weighted(xSMC, W, xSMC', epsilon[d])));
    statsSMC[j, :] .= mixture_hd_stats(xSMC, W, 1);
    # WGF
    tWGF[j] = @elapsed begin
    xWGF, _ = wgf_hd_mixture_tamed(Nparticles, dt, Niter, alpha[d], x0, m0, sigma0, muSample, sigmaK, false);
    end
    # entWGF[j] = mean(log.(mixture_hd_kde(xWGF, xWGF')));
    statsWGF[j, :] .= mixture_hd_stats(xWGF, ones(Nparticles)/Nparticles, 1);
    println("$d, $j")
end
statsSMC[:, 1:3] = (statsSMC[:, 1:3] .- [m v p]).^2;
statsWGF[:, 1:3] = (statsWGF[:, 1:3] .- [m v p]).^2;
open("1000smc_vs_wgf_$d.txt", "w") do io
    writedlm(io, [tSMC statsSMC tWGF statsWGF], ',')
end
