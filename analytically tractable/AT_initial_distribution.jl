push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using wgf;

# Compare initial distributions

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaK = 0.045^2;
sigmaRho = 0.043^2;
sigmaMu = sigmaRho + sigmaK;
rho(x) = pdf.(Normal(0.5, sqrt(sigmaRho)), x);
mu(x) = pdf.(Normal(0.5, sqrt(sigmaMu)), x);
K(x, y) = pdf.(Normal(x, sqrt(sigmaK)), y);

# dt and number of iterations
dt = 1e-03;
Niter = 1000;
# samples from h(y)
M = 500;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 500;
# regularisation parameter
alpha = 0.05;

# exact minimiser
variance, _  = AT_exact_minimiser(sigmaK, sigmaMu, alpha);
# initial distributions
x0 = [0.0*ones(1, Nparticles); 0.5*ones(1, Nparticles);
    1*ones(1, Nparticles); rand(1, Nparticles);
    0.5 .+ sqrt(variance)*randn(1, Nparticles);
    0.5 .+ sqrt(variance+0.01)*randn(1, Nparticles)];

E = zeros(Niter-1, size(x0, 1));
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = KDEx);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : zero(x)
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = mu.(refY);
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(K.(KDEx, refY[i]).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-alpha*ent;
end
for i=1:size(x0, 1)
    ### WGF
    x, _ = wgf_AT_tamed(Nparticles, dt, Niter, alpha, x0[i, :], M, 0.5);
    # KDE
    KDEyWGF = mapslices(phi, x[2:end, :], dims = 2);
    E[:, i] = mapslices(psi, KDEyWGF, dims = 2);
end

iterations = repeat(2:Niter, outer=[6, 1]);
# plot
R"""
    library(ggplot2)
    library(cowplot)
    library(scales)
    glabels <- c(expression(delta[0]), expression(delta[0.5]), expression(delta[1]),
        "U(0, 1)", expression(N(m, sigma[alpha]^2)), expression(N(m, sigma[alpha]^2+ epsilon)))
    g <- rep(1:6, , each= $Niter -1)
    data <- data.frame(x = $iterations, y = c($E), g = g);
    p <- ggplot(data, aes(x, y, group = factor(g), color = factor(g))) +
    geom_line(size = 2) +
    scale_colour_discrete(labels=glabels) +
    scale_y_log10(labels = trans_format("log10", math_format(10^.x))) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), legend.position="bottom", aspect.ratio = 2/4) +
    guides(colour = guide_legend(nrow = 1))
    ggsave("initial_distribution_E.eps", p, height=5)
"""
