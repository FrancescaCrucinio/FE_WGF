push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using Distances;
using XLSX;
using RCall;
@rimport ks as rks;
# custom packages
using wgf;

# set seed
Random.seed!(1234);

# sample from μ
muSample = float.(XLSX.readdata("sucrase_Carter1981.xlsx", "Foglio1!B2:B25"));
muSample = muSample[:];
# KDE for μ
bw = 1.06*std(muSample)*length(muSample)^(-1/5);
RKDE = rks.kde(muSample, var"h" = bw);
muKDEy = abs.(rcopy(RKDE[3]));
# reference values for KL divergence
muKDEx = rcopy(RKDE[2]);

a = 0.5;
alpha = 5;
Nparticles = 500;
dt = 1e-2;
Niter = 50000;
M = 500;
x0 = sample(muSample, Nparticles, replace = true);
x = wgf_sucrase_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, a);

# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t);
    return abs.(rcopy(RKDE[3]));
end
# function computing entropy
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : zero(x)
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    return ent;
end

KDEyWGF = mapslices(phi, x, dims = 2);
entWGF = mapslices(psi, KDEyWGF, dims = 2);

RKDE = rks.kde(x[Niter, :]);
KDEx = rcopy(RKDE[2]);
KDEy = abs.(rcopy(RKDE[3]));
# plot
R"""
    library(ggplot2)
    data <- data.frame(x = 1:$Niter, y = $entWGF)
    p1 <- ggplot(data, aes(x, y)) +
    geom_line(size = 2) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("sucrase_convergence.eps", p1,  height=5)
    g <- rep(1:2, , each= c(length($muKDEx), length($KDEx)));
    glabels <- c(expression(mu(y)), expression(rho(x)));
    data <- data.frame(x = c($muKDEx, $KDEx), y = c($muKDEy, $KDEy), g = g)
    p2 <- ggplot(data, aes(x, y, color = factor(g))) +
    geom_line(size = 2) +
    scale_colour_manual(values = c("red", "blue"), labels=glabels) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("sucrase.eps", p2,  height=5)
"""
