#push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
 push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
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
using samplers;

# set seed
Random.seed!(1234);

# data for anaytically tractable example
# data for gaussian mixture example
f(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
h(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
g(x, y) = pdf.(Normal(x, 0.045), y);

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from h(y)
M = 1000;
# number of particles
Nparticles = [100; 500; 1000; 5000; 10000];
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 100);
# regularisation parameters
epsilon = 1e-3;
alpha = 2e-2;
# number of repetitions
Nrep = 100;

# diagnostics
tSMC = zeros(length(Nparticles), 1);
diagnosticsSMC = zeros(length(Nparticles), 3);
qdistSMC = zeros(length(KDEx), length(Nparticles));
entropySMC = zeros(length(Nparticles), Nrep);
tWGF = zeros(length(Nparticles), 1);
diagnosticsWGF = zeros(length(Nparticles), 3);
qdistWGF = zeros(length(KDEx), length(Nparticles));
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
        x0WGF = 0.5*ones(1, Nparticles[i]);
        # sample from h(y)
        hSample = Ysample_gaussian_mixture(10);
        # run SMC
        trepSMC[j] = @elapsed begin
            xSMC, W = smc_gaussian_mixture(Nparticles[i], Niter, epsilon, x0SMC, hSample, M);
            # kde
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
            RKDESMC = rks.kde(x = xSMC[end,:], var"h" = bw, var"eval.points" = KDEx, var"w" = Nparticles[i]*W[end, :]);
            KDEySMC =  abs.(rcopy(RKDESMC[3]));
        end
        mSMC, vSMC, qSMC, miseSMC, eSMC = diagnosticsF(f, KDEx, KDEySMC);
        drepSMC[j, :] = [mSMC vSMC miseSMC];
        qdistrepSMC[j, :] = qSMC;
        entropySMC[i, j] = eSMC;
        # run WGF
        trepWGF[j] = @elapsed begin
            xWGF = wgf_gaussian_mixture_tamed(Nparticles[i], dt, Niter, alpha, x0WGF, hSample, M, 0.5);
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
    qdistSMC[:, i] = mean(qdistrepSMC, dims = 1);
    qdistWGF[:, i] = mean(qdistrepWGF, dims = 1);
end

miseSMC = diagnosticsSMC[:, 3];
miseWGF = diagnosticsWGF[:, 3];
groups = length(Nparticles);
# plot
R"""
    library(ggplot2)
    # mise vs runtime
    g <- rep(1:2, , each= $groups)
    symbol <- rep(c("N=100", "N=500", "N=1000", "N=5000", "N=10000"), times= 2)
    data <- data.frame(x = c($tSMC, $tWGF), y = c($miseSMC, $miseWGF), g = g);
    data$symbol <- factor(symbol, levels = c("N=100", "N=500", "N=1000", "N=5000", "N=10000"))
    p1 <- ggplot(data, aes(x, y, group = factor(g), color = factor(g))) +
    geom_line(size = 2) +
    geom_point(aes(shape=symbol), size=4) +
    scale_colour_manual(values = c("blue", "red"), labels=c("SMC-EMS", "WGF")) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("mixture_runtime_vs_mise.eps", p1,  height=5)

    # boxplot for smoothness
    symbol <- rep(c("N=100", "N=500", "N=1000", "N=5000", "N=10000", "N=100", "N=500", "N=1000", "N=5000", "N=10000"), each= 100)
    g <- rep(1:2, , each= $groups*100)
    runtime <- rep(c($tSMC, $tWGF), each = 100)
    runtime <- round(runtime, 2)
    data <- data.frame(x = factor(runtime), y = c(c($qdistSMC), c($qdistWGF)), g = g);
    data$symbol <- factor(symbol, levels = c("N=100", "N=500", "N=1000", "N=5000", "N=10000"))
    p2 <- ggplot(data) +
    geom_boxplot(aes(x = x, y=y, color = symbol)) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank())
    # ggsave("mixture_runtime_vs_mse.eps", p2,  height=5)
"""
#
save("smc_vs_wgf9Oct2020.jld", "alpha", alpha, "epsilon", epsilon, "diagnosticsWGF", diagnosticsWGF,
#     "diagnosticsSMC", diagnosticsSMC, "dt", dt, "tSMC", tSMC, "tWGF", tWGF,
#     "Nparticles", Nparticles, "Niter", Niter, "qdistWGF", qdistWGF,
#     "qdistSMC", qdistSMC);
#
#     Nparticles = load("smc_vs_wgf5Oct2020.jld", "Nparticles");
#     tSMC = load("smc_vs_wgf5Oct2020.jld", "tSMC");
#     tWGF = load("smc_vs_wgf5Oct2020.jld", "tWGF");
#     diagnosticsSMC = load("smc_vs_wgf5Oct2020.jld", "diagnosticsSMC");
#     diagnosticsWGF = load("smc_vs_wgf5Oct2020.jld", "diagnosticsWGF");
#     tSMC = load("smc_vs_wgf5Oct2020.jld", "tSMC");
#     tWGF = load("smc_vs_wgf5Oct2020.jld", "tWGF");
#     qdistSMC = load("smc_vs_wgf5Oct2020.jld", "qdistSMC");
#     qdistWGF = load("smc_vs_wgf5Oct2020.jld", "qdistWGF");
p1 = histogram(entropySMC[1, :]);
histogram!(entropyWGF[1, :]);
p2 = histogram(entropySMC[2, :]);
histogram!(entropyWGF[2, :]);
p3 = histogram(entropySMC[3, :]);
histogram!(entropyWGF[3, :]);
p4 = histogram(entropySMC[4, :]);
histogram!(entropyWGF[4, :]);
p5 = histogram(entropySMC[5, :]);
histogram!(entropyWGF[5, :]);
p6 = plot()
plot(p1, p2, p3, p4, p5, p6, layout=(3, 2))
