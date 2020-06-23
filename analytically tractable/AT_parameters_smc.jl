push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
# custom packages
using diagnostics;
using smcems;

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaF = 0.043^2;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);

# dt and number of iterations
Niter = 100;
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameters
epsilon = range(0.0, stop = 0.1, length = 10);
# number of repetitions
Nrep = 10;

function remove_non_finite(x)
       return isfinite(x) ? x : zero(x)
end

# diagnostics
diagnosticsSMC = zeros(length(epsilon), 1);
Threads.@threads for i=1:length(epsilon)
    # mise, mean and variance
    drepSMC = zeros(Nrep, 1);
    @simd for j=1:Nrep
        # initial distribution
        x0 = rand(1, Nparticles);
        # run WGF
        xSMC, W = smc_AT_approximated_potential(Nparticles, Niter, epsilon[i], x0, M);
        # kde
        bw = sqrt(epsilon[i]^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
        KDEySMC = weightedKDE(xSMC[end, :], W[end, :], bw, KDEx);
        drepSMC[j] = -mean(remove_non_finite.(KDEySMC .* log.(KDEySMC)));
        println("$i, $j")
    end
    diagnosticsSMC[i] = mean(drepSMC);
end
