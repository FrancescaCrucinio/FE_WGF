# push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
push!(LOAD_PATH, "/homes/crucinio/WGF/myModules")
# Julia packages
# using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD2;
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
M = 1000;
# number of particles
Nparticles = [100, 500, 1000, 5000];
# other parameters
### SMC
epsilon = 1e-03;
### WGF
lambda = 25;

# number of repetitions
Nrep = 1000;

# diagnostics
tSMC = zeros(length(Nparticles), 1);
diagnosticsSMC = zeros(length(Nparticles), 3);
tWGF = zeros(length(Nparticles), 1);
diagnosticsWGF = zeros(length(Nparticles), 3);
Threads.@threads for i=1:length(Nparticles)
    # times
    trepSMC = zeros(Nrep, 1);
    trepWGF = zeros(Nrep, 1);
    # mise, mean and variance
    drepSMC = zeros(Nrep, 3);
    drepWGF = zeros(Nrep, 3);
    @simd for j=1:Nrep
        println("$i, $j")
        # initial distribution
        x0 = rand(1, Nparticles[i]);
        # run SMC
        trepSMC[j] = @elapsed begin
             xSMC, W = smc_AT_approximated_potential(Nparticles[i], Niter, epsilon, x0, M);
            # kde
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
            KDEx = range(0, stop = 1, length = 1000);
            KDEySMC = weightedKDE(xSMC[end, :], W[end, :], bw, KDEx);
        end
        mSMC, vSMC, _, miseSMC, _ = diagnosticsF(f, KDEx, KDEySMC);
        drepSMC[j, :] = [mSMC, vSMC, miseSMC];
        # run WGF
        trepWGF[j] = @elapsed begin
            xWGF, drift = wgf_AT_approximated(Nparticles[i], Niter, lambda, x0, M);
            KDEyWGF = kerneldensity(xWGF[end, :], xeval = KDEx);
        end
        mWGF, vWGF, _, miseWGF, _ = diagnosticsF(f, KDEx, KDEyWGF);
        drepWGF[j, :] = [mWGF, vWGF, miseWGF];
    end
    tSMC[i] = mean(trepSMC);
    tWGF[i] = mean(trepWGF);
    diagnosticsSMC[i, :] = mean(drepSMC, dims = 1);
    diagnosticsWGF[i, :] = mean(drepWGF,dims = 1);
end

# p1 = plot(Nparticles, [tSMC, tWGF], lw = 3, xlabel="N", ylabel="Runtime",
#         label = ["SMC" "WGF"], legend=:topleft);
# p2 = plot(Nparticles, [diagnosticsSMC[:, 1], diagnosticsWGF[:, 1]],
#     lw = 3, xlabel="N", ylabel="mean", legend = false);
# hline!([0.5]);
# p3 = plot(Nparticles, [diagnosticsSMC[:, 2], diagnosticsWGF[:, 2]],
#     lw = 3, legend = false, xlabel="N", ylabel="variance");
# hline!([0.043^2]);
# p4 = plot(Nparticles, [diagnosticsSMC[:, 3], diagnosticsWGF[:, 3]],
#     lw = 3, legend = false, xlabel="N", ylabel="MISE");
# plot(p1, p2, p3, p4, layout = (2, 2))
#
# savefig(p1, "comparison_runtime.pdf")
# savefig(p2, "comparison_mean.pdf")
# savefig(p3, "comparison_var.pdf")
# savefig(p4, "comparison_mise.pdf")

@save "comparison.jld"
