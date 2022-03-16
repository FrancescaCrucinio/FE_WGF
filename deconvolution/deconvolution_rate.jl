# push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
push!(LOAD_PATH, "/Users/francescacrucinio/Documents/WGF/myModules")
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
# number of particles
Nparticles = [100; 500; 1000; 5000; 10000];
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 100);
dx = KDEx[2] - KDEx[1];
true_density = pi(KDEx);
# regularisation parameters
epsilon = [4.5e-4 3.3e-4 1.1e-3 3.3e-3 3.8e-3];
alpha = [4.5e-4 3.3e-4 1.1e-3 3.3e-3 3.8e-3];
# alpha = [1.1e-1 5e-2 6e-2 3.1e-2 2.1e-2];
# number of repetitions
Nrep = 100;

# diagnostics
tPI = zeros(length(Nparticles), 1);
isePI = zeros(length(Nparticles), Nrep);
qdistPI = zeros(length(KDEx), length(Nparticles));
tCV = zeros(length(Nparticles), 1);
iseCV = zeros(length(Nparticles), 1);
var_iseCV = zeros(length(Nparticles), 1);
qdistCV = zeros(length(KDEx), length(Nparticles));
tSMC = zeros(length(Nparticles), 1);
iseSMC = zeros(length(Nparticles), Nrep);
qdistSMC = zeros(length(KDEx), length(Nparticles));
tWGF = zeros(length(Nparticles), 1);
iseWGF = zeros(length(Nparticles), Nrep);
var_iseWGF = zeros(length(Nparticles), 1);
qdistWGF = zeros(length(KDEx), length(Nparticles));

for i=1:length(Nparticles)
    # times
    trepPI = zeros(Nrep, 1);
    trepSMC = zeros(Nrep, 1);
    trepWGF = zeros(Nrep, 1);
    # mse
    qdistrepPI = zeros(Nrep, length(KDEx));
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
        """
        # runtimes and ise
        trepPI[j] = @rget exectimePI;
        qdistrepPI[j, :] = (true_density .- @rget(fdec_hPI)).^2;
        isePI[i, j] = dx*sum(qdistrepPI[j, :]);

        # SMC
        # initial distribution
        x0 = sample(muSample, Nparticles[i], replace = true);
        M = min(Nparticles[i], length(muSample));
        trepSMC[j] = @elapsed begin
            xSMC, W = smc_gaussian_mixture(Nparticles[i], Niter, epsilon[i], x0, muSample, M);
            # kde
            bw = sqrt(epsilon[i]^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
            RKDESMC = rks.kde(x = xSMC[end,:], var"h" = bw, var"eval.points" = KDEx, var"w" = Nparticles[i]*W[end, :]);
            KDEySMC =  abs.(rcopy(RKDESMC[3]));
        end
        qdistrepSMC[j, :] = (true_density .- KDEySMC).^2;
        iseSMC[i, j] = dx*sum(qdistrepSMC[j, :]);

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
        iseWGF[i, j] = dx*sum(qdistrepWGF[j, :]);
        println("$i, $j")
    end
    tPI[i] = mean(trepPI);
    # tCV[i] = mean(trepCV);
    tSMC[i] = mean(trepSMC);
    tWGF[i] = mean(trepWGF);
    qdistPI[:, i] = mean(qdistrepPI, dims = 1);
    # qdistCV[:, i] = mean(qdistrepCV, dims = 1);
    qdistSMC[:, i] = mean(qdistrepSMC, dims = 1);
    qdistWGF[:, i] = mean(qdistrepWGF, dims = 1);
end

var_isePI = var(isePI, dims = 2);
var_iseSMC = var(iseSMC, dims = 2);
var_iseWGF = var(iseWGF, dims = 2);
p = plot(tPI, mean(isePI, dims = 2), xaxis = :log, lw = 3, color = :orange, line = :dashdotdot, label = "DKDE-pi",
    legendfontsize = 12, tickfontsize = 10, legend = :outerright, size=(800, 400), ribbon = sqrt.(var_isePI), fillalpha = .2,
    xlim = (minimum(tWGF), maximum(tSMC)+35), xticks = [1e-01, 1e+00, 1e+01, 1e+02])
plot!(p, tSMC, mean(iseSMC, dims = 2), xaxis = :log, lw = 3, color = :purple, line = :dash,  label = "SMC-EMS", ribbon =  sqrt.(var_iseSMC), fillalpha = .2)
plot!(p, tWGF, mean(iseWGF, dims = 2), xaxis = :log, lw = 3, color = :red, line = :solid, label = "Algo 1", ribbon = sqrt.(var_iseWGF), fillalpha = .2)
markers = [:circle :rect :diamond :star5 :xcross];
for i=1:length(Nparticles)
    N = Nparticles[i];
    scatter!(p, [tPI[i]], [mean(isePI, dims = 2)[i]], xaxis = :log, color = :black,
        markerstrokecolor = :black, marker = markers[i], markersize = 5, label = "N = $N")
    scatter!(p, [tPI[i]], [mean(isePI, dims = 2)[i]], xaxis = :log, color = :orange,
        markerstrokecolor = :orange, marker = markers[i], markersize = 5, label = "")
    scatter!(p, [tSMC[i]], [mean(iseSMC, dims = 2)[i]], xaxis = :log, color = :purple,
        markerstrokecolor = :purple, marker = markers[i], markersize = 5, label = "")
    scatter!(p, [tWGF[i]], [mean(iseWGF, dims = 2)[i]], xaxis = :log, color = :red,
        markerstrokecolor = :red, marker = markers[i], markersize = 5, label = "")
end
# savefig(p, "mixture_runtime_vs_mise.pdf")
bp1 = boxplot(transpose(log10.(tPI)), qdistPI, yaxis = :log10, legend = :none, bar_width = 0.2,
tickfontsize = 15, ylims = (0.5*minimum(qdistSMC), maximum(qdistPI)), xlims = (minimum(log10.(tSMC))-0.5, maximum(log10.(tSMC))+0.5))
# title = "DKDE-pi", ylabel = "MSE", xlabel = "Runtime (log s)")
# savefig(bp1, "mixture_runtime_vs_mse_pi.pdf")
bp3 = boxplot(transpose(log10.(tSMC)), qdistSMC, yaxis = :log10, legend = :none, bar_width = 0.3, range = 0,
tickfontsize = 15, ylims = (0.5*minimum(qdistSMC), maximum(qdistPI)), xlims = (minimum(log10.(tSMC))-0.5, maximum(log10.(tSMC))+0.5))
# title = "SMC-EMS", ylabel = "MSE", xlabel = "Runtime (log s)")
# savefig(bp3, "mixture_runtime_vs_mse_smc.pdf")
bp4 = boxplot(transpose(log10.(tWGF)), qdistWGF, yaxis = :log10, legend = :none, bar_width = 0.3, range = 0,
tickfontsize = 15, ylims = (0.5*minimum(qdistSMC), maximum(qdistPI)), xlims = (minimum(log10.(tSMC))-0.5, maximum(log10.(tSMC))+0.5))
# title = "WGF", ylabel = "MSE", xlabel = "Runtime (log s)")
# savefig(bp4, "mixture_runtime_vs_mse_wgf.pdf")
legend = scatter([0 0 0 0 0], showaxis = false, grid = false, label = ["N = 100" "N = 500" "N=1000" "N=5000" "N=10000"],
    markerstrokewidth = 0, markersize = 10, fontsize = 20, legend = :outertopright, size = (100, 200))
scatter!(legend, [0], markercolor = :white, label = "", markerstrokecolor = :white, markersize = 13)
# savefig(legend, "mixture_runtime_vs_mse_legend.pdf")
bp = plot(bp1, bp3, bp4, legend, layout = @layout([[A B C] E{.15w}]), size = (900, 400), tickfontsize = 10)
# savefig(bp, "mixture_runtime_vs_mse.pdf")

# save("prior_deconv_rate24December2021.jld", "tPI", tPI, "tSMC", tSMC, "tWGF", tWGF,
#       "isePI", isePI, "iseSMC", iseSMC, "iseWGF", iseWGF,
#       "qdistPI", qdistPI, "qdistSMC", qdistSMC, "qdistWGF", qdistWGF);

tPI = load("data/prior_deconv_rate14March2022.jld", "tPI");
tSMC = load("data/prior_deconv_rate14March2022.jld", "tSMC");
tWGF = load("data/prior_deconv_rate14March2022.jld", "tWGF");
isePI = load("data/prior_deconv_rate14March2022.jld", "isePI");
iseSMC = load("data/prior_deconv_rate14March2022.jld", "iseSMC");
iseWGF = load("data/prior_deconv_rate14March2022.jld", "iseWGF");
qdistPI = load("data/prior_deconv_rate14March2022.jld", "qdistPI");
qdistSMC = load("data/prior_deconv_rate14March2022.jld", "qdistSMC");
qdistWGF = load("data/prior_deconv_rate14March2022.jld", "qdistWGF");

R"""
library(ggplot2)
df <- data.frame(c($qdistPI, $qdistSMC, $qdistWGF), rep(c($tPI, $tSMC, $tWGF), each = 100),
    rep($Nparticles, each = 100), rep(c("DKDEpi", "SMCEMS", "WGF"), each = 5*100))
colnames(df) <- c("qdist", "t", "N", "algo")
df$N <- as.factor(df$N)
distances_mean <- data.frame(aggregate(qdist ~ algo + N, data = df, FUN = "mean"))
time_means <- aggregate(t ~ algo + N, data = df, FUN= "mean" )
distances_mean <- merge(distances_mean, time_means, by=c("algo", "N"))
ggplot(data = df, aes(x = t, y = qdist, group = N, colour = N, fill = N)) +
    geom_boxplot(width = 0.2, alpha = 0.2, lwd = 1, coef = 6) +
#    geom_point(data = distances_mean, shape = 4, lwd = 10, aes(x = t, y = qdist, group = interaction(algo, N), fill = N, colour = N)) +
    scale_x_log10(
        breaks = scales::trans_breaks("log10", function(x) 10^x),
        labels = scales::trans_format("log10", scales::math_format(10^.x))
    ) +
    scale_y_log10(
        breaks = scales::trans_breaks("log10", function(x) 10^x),
        labels = scales::trans_format("log10", scales::math_format(10^.x))
        ) +
    facet_wrap(~algo) +
    theme_bw() +
    theme(axis.title.x=element_blank(), axis.title.y=element_blank(),
        legend.title = element_blank(), legend.text=element_text(size=25),
        text = element_text(size=20), legend.position="right")
ggsave("mixture_runtime_vs_mse.pdf", width = 14, height = 8, dpi = 300)
"""
