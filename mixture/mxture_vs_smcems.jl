push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
# using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
using RCall;
@rimport ks as rks
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

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 1000;
# number of particles
Nparticles = [100; 500; 1000; 5000; 10000];
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# regularisation parameters
epsilon = 0.001;
alpha = 0.01; # chosen to give the same entropy as SMCEMS
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
        x0SMC = rand(1, Nparticles[i]);
        x0SMC = rand(1, Nparticles[i]);#????
        # run SMC
        trepSMC[j] = @elapsed begin
            xSMC, W = smc_AT_approximated_potential(Nparticles[i], Niter, epsilon, x0, M); ###write smc for gaussian mixture
            # kde
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
            RKDESMC = rks.kde(x = xSMC[end,:], var"h" = bw, var"eval.points" = KDEx, var"w" = W[end, :]);
            KDEySMC =  abs.(rcopy(RKDESMC[3]));
        end
        mSMC, vSMC, qSMC, miseSMC, eSMC = diagnosticsF(f, KDEx, KDEySMC);
        drepSMC[j, :] = [mSMC vSMC miseSMC];
        qdistrepSMC[j, :] = qSMC;
        entropySMC[i, j] = eSMC;
        # run WGF
        trepWGF[j] = @elapsed begin
            xWGF = wgf_gaussian_mixture(Nparticles[i], dt, Niter, alpha, x0WGF, M);
            RKDEWGF = rks.kde(x = xWGF[end,:], var"eval.points" = KDEx);
            KDEyWGF =  abs.(rcopy(RKDEWGF[3]));
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
