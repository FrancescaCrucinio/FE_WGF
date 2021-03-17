push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using RCall;
using JLD;
@rimport ks as rks;
# custom packages
using wgf_prior;
using samplers;
using smcems;
R"""
library(fDKDE)
library(tictoc)
memory.limit(17000)
"""

# set seed
Random.seed!(1234);

# data for gaussian mixture example
pi(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
mu(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
K(x, y) = pdf.(Normal(x, 0.045), y);
sdK = 0.045;

# parameters
# dt and number of iterations
dt = 1e-03;
Niter = 100;
# samples from μ(y)
M = 1000;
# number of particles
Nparticles = [100; 500; 1000; 5000; 10000];
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 100);
dx = KDEx[2] - KDEx[1];
true_density = pi(KDEx);
# regularisation parameters
epsilon = 1e-3;
alpha = 2e-2;
# number of repetitions
Nrep = 10;

# diagnostics
tPI = zeros(length(Nparticles), 1);
isePI = zeros(length(Nparticles), 1);
qdistPI = zeros(length(KDEx), length(Nparticles));
tCV = zeros(length(Nparticles), 1);
iseCV = zeros(length(Nparticles), 1);
qdistCV = zeros(length(KDEx), length(Nparticles));
tSMC = zeros(length(Nparticles), 1);
iseSMC = zeros(length(Nparticles), 1);
qdistSMC = zeros(length(KDEx), length(Nparticles));
tWGF = zeros(length(Nparticles), 1);
iseWGF = zeros(length(Nparticles), 1);
qdistWGF = zeros(length(KDEx), length(Nparticles));

for i=1:length(Nparticles)
    # times
    trepPI = zeros(Nrep, 1);
    trepCV = zeros(Nrep, 1);
    trepSMC = zeros(Nrep, 1);
    trepWGF = zeros(Nrep, 1);
    # ise
    iserepPI = zeros(Nrep, 1);
    iserepCV = zeros(Nrep, 1);
    iserepSMC = zeros(Nrep, 1);
    iserepWGF = zeros(Nrep, 1);
    # mse
    qdistrepPI = zeros(Nrep, length(KDEx));
    qdistrepCV = zeros(Nrep, length(KDEx));
    qdistrepWGF = zeros(Nrep, length(KDEx));
    qdistrepSMC = zeros(Nrep, length(KDEx));
    for j=1:Nrep
        # sample from μ(y)
        muSample = Ysample_gaussian_mixture(10^3);
        muSampleDKDE = Ysample_gaussian_mixture(Nparticles[i]);
        # DKDEpi & DKDEcv
        R"""
        # PI bandwidth of Delaigle and Gijbels
        tic()
        hPI <- PI_deconvUknownth4(c($muSampleDKDE), "norm", $sdK^2, $sdK);
        fdec_hPI <- fdecUknown($KDEx, c($muSampleDKDE), hPI, "norm", $sdK, $dx);
        exectime <- toc()
        exectimePI <- exectime$toc - exectime$tic
        tic()
        hCV <- CVdeconv(c($muSampleDKDE), "norm", $sdK);
        fdec_hCV <-  fdecUknown($KDEx, c($muSampleDKDE), hCV, "norm", $sdK, $dx);
        exectime <- toc()
        exectimeCV <- exectime$toc - exectime$tic
        """
        # runtimes and ise
        trepPI[j] = @rget exectimePI;
        trepCV[j] = @rget exectimeCV;
        qdistrepPI[j, :] = (true_density .- @rget(fdec_hPI)).^2;
        qdistrepCV[j, :] = (true_density .- @rget(fdec_hCV)).^2;
        iserepPI[j] = dx*sum(qdistrepPI[j, :]);
        iserepCV[j] = dx*sum(qdistrepCV[j, :]);

        # SMC
        # initial distribution
        x0SMC = rand(1, Nparticles[i]);
        M = min(Nparticles[i], length(muSample));
        trepSMC[j] = @elapsed begin
            xSMC, W = smc_gaussian_mixture(Nparticles[i], Niter, epsilon, x0SMC, muSample, M);
            # kde
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
            RKDESMC = rks.kde(x = xSMC[end,:], var"h" = bw, var"eval.points" = KDEx, var"w" = Nparticles[i]*W[end, :]);
            KDEySMC =  abs.(rcopy(RKDESMC[3]));
        end
        qdistrepSMC[j, :] = (true_density .- KDEySMC).^2;
        iserepSMC[j] = dx*sum(qdistrepSMC[j, :]);
        # WGF
        # prior mean = mean of μ
        m0 = mean(muSample);
        sigma0 = std(muSample);
        # initial distribution
        x0WGF = sample(muSample, Nparticles[i], replace = true);
        M = min(Nparticles[i], length(muSample));
        trepWGF[j] = @elapsed begin
        x = wgf_DKDE_tamed(Nparticles[i], dt, Niter, alpha, x0WGF, m0, sigma0, muSample, M, sdK);
        RKDEyWGF = rks.kde(x = x[Niter, :], var"eval.points" = KDEx);
        KDEyWGF = abs.(rcopy(RKDEyWGF[3]));
        end
        qdistrepWGF[j, :] = (true_density .- KDEyWGF).^2;
        iserepWGF[j] = dx*sum(qdistrepWGF[j, :]);
        println("$i, $j")
    end
    tPI[i] = mean(trepPI);
    tCV[i] = mean(trepCV);
    tSMC[i] = mean(trepSMC);
    tWGF[i] = mean(trepWGF);
    isePI[i] = mean(iserepPI);
    iseCV[i] = mean(iserepCV);
    iseSMC[i] = mean(iserepSMC);
    iseWGF[i] = mean(iserepWGF);
    qdistPI[:, i] = mean(qdistrepPI, dims = 1);
    qdistCV[:, i] = mean(qdistrepCV, dims = 1);
    qdistSMC[:, i] = mean(qdistrepSMC, dims = 1);
    qdistWGF[:, i] = mean(qdistrepWGF, dims = 1);
end

p = plot(tPI, isePI, xaxis = :log, lw = 3, color = :orange, line = :dashdotdot, label = "DKDEpi",
    legendfontsize = 10, tickfontsize = 10, legend = :outerright)
plot!(p, tCV, iseCV, xaxis = :log, lw = 3, color = :blue, line = :dot,  label = "DKDEcv")
plot!(p, tSMC, iseSMC, xaxis = :log, lw = 3, color = :purple, line = :dash,  label = "SMC-EMS")
plot!(p, tWGF, iseWGF, xaxis = :log, lw = 3, color = :red, line = :solid, label = "WGF")
markers = [:circle :rect :diamond :star5 :xcross];
for i=1:length(Nparticles)
    N = Nparticles[i];
    scatter!(p, [tPI[i]], [isePI[i]], xaxis = :log, color = :black,
        markerstrokecolor = :black, marker = markers[i], markersize = 5, label = "N = $N")
    scatter!(p, [tPI[i]], [isePI[i]], xaxis = :log, color = :orange,
        markerstrokecolor = :orange, marker = markers[i], markersize = 5, label = "")
    scatter!(p, [tCV[i]], [iseCV[i]], xaxis = :log, color = :blue,
        markerstrokecolor = :blue, marker = markers[i], markersize = 5, label = "")
    scatter!(p, [tSMC[i]], [iseSMC[i]], xaxis = :log, color = :purple,
        markerstrokecolor = :purple, marker = markers[i], markersize = 5, label = "")
    scatter!(p, [tWGF[i]], [iseWGF[i]], xaxis = :log, color = :red,
        markerstrokecolor = :red, marker = markers[i], markersize = 5, label = "")
end

#
save("deconv_rate16Mar2021.jld", "tPI", tPI, "tCV", tCV, "tWGF", tWGF, "tSMC", tSMC,
     "iseWGF", iseWGF, "iseCV", iseCV, "isePI", isePI, "iseSMC", iseSMC,
     "qdistPI", qdistPI, "qdistCV", qdistCV, "qdistSMC", qdistSMC, "qdistWGF", qdistWGF);

bp1 = boxplot(transpose(tPI), qdistPI, yaxis = :log, legend = :none)
bp2 = boxplot(transpose(tCV), qdistCV, yaxis = :log, legend = :none)
bp3 = boxplot(transpose(tSMC), qdistSMC, yaxis = :log, legend = :none)
bp4 = boxplot(transpose(tWGF), qdistWGF, yaxis = :log, legend = :none)
plot(bp1, bp2, bp3, bp4, layout = (2, 2))
