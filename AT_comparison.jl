push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
# custom packages
using diagnostics;
using smcems;
using wgf;

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
h(x) = pdf.(Normal(0.5, sqrt(sigmaH)), x);
g(x, y) = pdf.(Normal(x, sqrt(sigmaG)), y);

# common parameters
# number of iterations
Niter = trunc(Int, 1e03);
# samples from h(y)
M = 100;
# initial distribution
x0 = rand(1, N);
# number of particles
Nparticles = [100, 500];
# other parameters
### SMC
epsilon = 1e-03;
### WGF
lambda = 30;

# number of repetitions
Nrep = 5;

# diagnostics
tSMC = zeros(length(Nparticles), 1);
diagnosticsSMC = zeros(length(Nparticles), 3);
tWGF = zeros(length(Nparticles), 1);
diagnosticsWGF = zeros(length(Nparticles), 3);
for i=1:length(Nparticles)
    # times
    trepSMC = zeros(Nrep, 1);
    trepWGF = zeros(Nrep, 1);
    # mise, mean and variance
    drepSMC = zeros(Nrep, 3);
    drepWGF = zeros(Nrep, 3);
    for j=1:Nrep
        # run SMC
        trepSMC[j] = @elapsed begin
             xSMC, W = smc_AT_approximated_potential(Nparticles[i], Niter, epsilon, x0, M);
            # kde
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
            KDEx = range(0, stop = 1, length = 1000);
            KDEySMC = weightedKDE(xSMC[end, :], W[end, :], bw, KDEx);
        end
        drepSMC[j, :] = diagnosticsF(f, KDEx, KDEySMC);

        # run WGF
        trepWGF[j] = @elapsed begin
            xWGF, drift = wgf_AT_approximated(Nparticles[i], Niter, lambda, x0, M);
            KDEyWGF = kerneldensity(xWGF[end, :], xeval = KDEx);
        end
        drepWGF[j, :] = diagnosticsF(f, KDEx, KDEyWGF);
    end
    tSMC[i] = mean(trepSMC, 1);
    tWGF[i] = mean(trepWGF, 1);
    diagnosticsSMC[i, :] = mean(drepSMC, 1);
    diagnosticsWGF[i, :] = mean(drepWGF, 1);
end
