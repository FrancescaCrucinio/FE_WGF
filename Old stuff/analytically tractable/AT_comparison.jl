#push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# push!(LOAD_PATH, "/homes/crucinio/WGF/myModules")
# Julia packages
# using Revise;
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
# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 1000;
# number of particles
Nparticles = [100; 500; 1000; 5000; 10000];
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# other parameters
### SMC
# entropy of solution = -1.73
epsilon = 3e-09;
# entropy = -1
epsilon = 0.05;
# fix epsilon/ entropy = -1.35
epsilon = 1e-03;
### WGF
# entropy of solution = -1.73
lambda = 5e-04;
# entropy = -1
lambda = 0.29;
# fix epsilon/ entropy = -1.35
lambda = 0.125;
# number of repetitions
Nrep = 1000;

# diagnostics
tSMC = zeros(length(Nparticles), 1);
diagnosticsSMC = zeros(length(Nparticles), 3);
qdistSMC = zeros(length(Nparticles), length(KDEx));
entropySMC = zeros(length(Nparticles), Nrep);
tWGF = zeros(length(Nparticles), 1);
diagnosticsWGF = zeros(length(Nparticles), 3);
qdistWGF = zeros(length(Nparticles), length(KDEx));
entropyWGF = zeros(length(Nparticles), Nrep);
Threads.@threads for i=1:length(Nparticles)
    # times
    trepSMC = zeros(Nrep, 1);
    trepWGF = zeros(Nrep, 1);
    # mise, mean and variance
    drepSMC = zeros(Nrep, 3);
    drepWGF = zeros(Nrep, 3);
    qdistrepWGF = zeros(Nrep, length(KDEx));
    qdistrepSMC = zeros(Nrep, length(KDEx));
    @simd for j=1:Nrep
        # initial distribution
        x0 = rand(1)*ones(1, Nparticles[i]);
        # run SMC
        trepSMC[j] = @elapsed begin
            xSMC, W = smc_AT_approximated_potential(Nparticles[i], Niter, epsilon, x0, M);
            # kde
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
            KDEySMC = weightedKDE(xSMC[end, :], W[end, :], bw, KDEx);
        end
        mSMC, vSMC, qSMC, miseSMC, eSMC = diagnosticsF(f, KDEx, KDEySMC);
        drepSMC[j, :] = [mSMC vSMC miseSMC];
        qdistrepSMC[j, :] = qSMC;
        entropySMC[i, j] = eSMC;
        # run WGF
        trepWGF[j] = @elapsed begin
            xWGF, drift = wgf_AT(Nparticles[i], dt, Niter, lambda, x0, M);
            KDEyWGF =  KernelEstimator.kerneldensity(xWGF[end,:], xeval=KDEx, h=bwnormal(xWGF[end,:]));
        end
        mWGF, vWGF, qWGF, miseWGF, eWGF = diagnosticsF(f, KDEx, KDEyWGF);
        drepWGF[j, :] = [mWGF vWGF miseWGF];
        qdistrepWGF[j, :] = qWGF;
        entropyWGF[i, j] = eWGF;
        println("$i, $j")
    end
    tSMC[i] = mean(trepSMC);
    tWGF[i] = mean(trepWGF);
    diagnosticsSMC[i, :] = mean(drepSMC, dims = 1);
    diagnosticsWGF[i, :] = mean(drepWGF,dims = 1);
    qdistSMC[i, :] = mean(qdistrepSMC, dims = 1);
    qdistWGF[i, :] = mean(qdistrepWGF, dims = 1);
end


save("comparison_uniform21062020.jld", "lambda", lambda, "diagnosticsWGF", diagnosticsWGF,
    "diagnosticsSMC", diagnosticsSMC, "dt", dt, "tSMC", tSMC, "tWGF", tWGF,
    "Nparticles", Nparticles, "Niter", Niter, "qdistWGF", qdistWGF,
    "qdistSMC", qdistSMC);
