push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
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

function remove_non_finite(x)
       return isfinite(x) ? x : 0
end


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
# number of particles
Nparticles = [100; 500; 1000; 5000; 10000];
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 100);
dx = KDEx[2] - KDEx[1];
true_density = pi(KDEx);
# regularisation parameters
epsilon = 1e-3;
alpha = [1.1e-1 5e-2 6e-2 3.1e-2 2.1e-2];
# number of repetitions
Nrep = 100;

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
entSMC = zeros(length(Nparticles), Nrep);
entWGF = zeros(length(Nparticles), Nrep);

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
        x0 = sample(muSample, Nparticles[i], replace = true);
        M = min(Nparticles[i], length(muSample));
        trepSMC[j] = @elapsed begin
            xSMC, W = smc_gaussian_mixture(Nparticles[i], Niter, epsilon, x0, muSample, M);
            # kde
            bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
            RKDESMC = rks.kde(x = xSMC[end,:], var"h" = bw, var"eval.points" = KDEx, var"w" = Nparticles[i]*W[end, :]);
            KDEySMC =  abs.(rcopy(RKDESMC[3]));
        end
        qdistrepSMC[j, :] = (true_density .- KDEySMC).^2;
        iserepSMC[j] = dx*sum(qdistrepSMC[j, :]);
        entSMC[i, j] = -dx*sum(remove_non_finite.(KDEySMC .* log.(KDEySMC)));

        # WGF
        # prior mean = mean of μ
        m0 = mean(muSample);
        sigma0 = std(muSample);
        M = min(Nparticles[i], length(muSample));
        trepWGF[j] = @elapsed begin
        x = wgf_DKDE_tamed(Nparticles[i], dt, Niter, alpha[i], x0, m0, sigma0, muSample, M, sdK);
        RKDEyWGF = rks.kde(x = x[Niter, :], var"eval.points" = KDEx);
        KDEyWGF = abs.(rcopy(RKDEyWGF[3]));
        end
        qdistrepWGF[j, :] = (true_density .- KDEyWGF).^2;
        iserepWGF[j] = dx*sum(qdistrepWGF[j, :]);
        entWGF[i, j] = -dx*sum(remove_non_finite.(KDEyWGF .* log.(KDEyWGF)));
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


p = plot(tPI, isePI, xaxis = :log, lw = 3, color = :orange, line = :dashdotdot, label = "DKDE-pi",
    legendfontsize = 10, tickfontsize = 10, legend = :outerright, size=(700, 400))
plot!(p, tCV, iseCV, xaxis = :log, lw = 3, color = :blue, line = :dot,  label = "DKDE-cv")
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
# savefig(p, "mixture_runtime_vs_mise.pdf")
bp1 = boxplot(transpose(log10.(tPI)), qdistPI, yaxis = :log10, legend = :none, bar_width = 0.2, range = 0, tickfontsize = 15)
# title = "DKDE-pi", ylabel = "MSE", xlabel = "Runtime (log s)")
# savefig(bp1, "mixture_runtime_vs_mse_pi.pdf")
bp2 = boxplot(transpose(log10.(tCV)), qdistCV, yaxis = :log10, legend = :none, bar_width = 0.2, range = 0, tickfontsize = 15)
# title = "DKDE-cv", ylabel = "MSE", xlabel = "Runtime (log s)")
# savefig(bp2, "mixture_runtime_vs_mse_cv.pdf")
bp3 = boxplot(transpose(log10.(tSMC)), qdistSMC, yaxis = :log10, legend = :none, bar_width = 0.3, range = 0, tickfontsize = 15)
# title = "SMC-EMS", ylabel = "MSE", xlabel = "Runtime (log s)")
# savefig(bp3, "mixture_runtime_vs_mse_smc.pdf")
bp4 = boxplot(transpose(log10.(tWGF)), qdistWGF, yaxis = :log10, legend = :none, bar_width = 0.3, range = 0, tickfontsize = 15)
# title = "WGF", ylabel = "MSE", xlabel = "Runtime (log s)")
# savefig(bp4, "mixture_runtime_vs_mse_wgf.pdf")
legend = scatter([0 0 0 0 0], showaxis = false, grid = false, label = ["N = 100" "N = 500" "N=1000" "N=5000" "N=10000"],
    markerstrokewidth = 0, markersize = 10, fontsize = 20, legend = :outertopright, size = (100, 200))
scatter!(legend, [0], markercolor = :white, label = "", markerstrokecolor = :white, markersize = 13)
# savefig(legend, "mixture_runtime_vs_mse_legend.pdf")
bp = plot(bp1, bp2, bp3, bp4, legend, layout = @layout([[A B; C D] E{.15w}]), size = (900, 500), tickfontsize = 10)
# savefig(bp, "mixture_runtime_vs_mse.pdf")
#
save("prior_deconv_rate3Apr2021.jld", "tPI", tPI,  "tCV", tCV, "tSMC", tSMC, "tWGF", tWGF,
      "isePI", isePI,  "iseCV", iseCV, "iseSMC", iseSMC, "iseWGF", iseWGF,
      "entSMC", entSMC, "entWGF", entWGF,
      "qdistPI", qdistPI,  "qdistCV", qdistCV, "qdistSMC", qdistSMC, "qdistWGF", qdistWGF);

# # tPI = load("deconv_rate28Mar2021.jld", "tPI");
# # tCV = load("deconv_rate28Mar2021.jld", "tCV");
# # tSMC = load("deconv_rate28Mar2021.jld", "tSMC");
# # tWGF = load("deconv_rate28Mar2021.jld", "tWGF");
# # isePI = load("deconv_rate28Mar2021.jld", "isePI");
# # iseCV = load("deconv_rate28Mar2021.jld", "iseCV");
# # iseSMC = load("deconv_rate28Mar2021.jld", "iseSMC");
# # iseWGF = load("deconv_rate28Mar2021.jld", "iseWGF");
# # qdistPI = load("deconv_rate28Mar2021.jld", "qdistPI");
# # qdistCV = load("deconv_rate28Mar2021.jld", "qdistCV");
# # qdistSMC = load("deconv_rate28Mar2021.jld", "qdistSMC");
# # qdistWGF = load("deconv_rate28Mar2021.jld", "qdistWGF");
# # entSMC = load("deconv_rate28Mar2021.jld", "entSMC");
# # entWGF = load("deconv_rate28Mar2021.jld", "entWGF");
#
# i = 3
# histogram(entWGF[i, :])
# histogram!(entSMC[i, :])
