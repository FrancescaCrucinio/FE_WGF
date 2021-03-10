push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using diagnostics;
using wgf;
using samplers;

# set seed
Random.seed!(1234);

# data for gaussian mixture example
rho(x) = pdf.(Normal(0.3, 0.015), x)/3 + 2*pdf.(Normal(0.5, 0.043), x)/3;
mu(x) = 2*pdf.(Normal(0.3, sqrt(0.043^2 + 0.045^2)), x)/3 +
        pdf.(Normal(0.5, sqrt(0.015^2 + 0.045^2)), x)/3;
K(x, y) = pdf.(Normal(x, 0.045), y);

# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);

# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = KDEx);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
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
    return kl-a*ent;
end

# dt and number of iterations
dt = 1e-03;
Niter = 200;
# samples from h(y)
M = 1000;
# sample from h(y)
muSample = Ysample_gaussian_mixture(100000);
# number of particles
Nparticles = 1000;
# regularisation parameter
alpha = [0.01 0.1 0.5 1];
a = alpha[1];
E = zeros(Niter, length(alpha));
rho_approx = zeros(length(KDEx), length(alpha));
for i=1:length(alpha)
    x0 = 0.5 .+ randn(1, Nparticles)/10;
    # run WGF
    x =  wgf_gaussian_mixture_tamed(Nparticles, dt, Niter, alpha[i], x0, muSample, M, 0.5);
    a = alpha[i];
    KDEyWGF = mapslices(phi, x, dims = 2);
    rho_approx[:, i] = KDEyWGF[end, :];
    E[:, i] = mapslices(psi, KDEyWGF, dims = 2);
end

iterations = repeat(1:Niter, outer=[4, 1]);
solution = rho.(KDEx);
# plot
R"""
    library(ggplot2)
    glabels <- c(expression(paste(alpha, "=", 0.01)), expression(paste(alpha, "=", 0.1)),
        expression(paste(alpha, "=", 0.5)), expression(paste(alpha, "=", 1)));
    g <- rep(1:4, , each= $Niter)
    data <- data.frame(x = $iterations, y = c($E), g = g);
    p1 <- ggplot(data, aes(x, y, group = factor(g), color = factor(g))) +
    geom_line(size = 2) +
    scale_colour_discrete(labels=glabels) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    x = rep($KDEx, 5);
    g <- rep(1:5, , each= length($KDEx));
    glabels <- c(expression(rho(x)), glabels);
    data <- data.frame(x = x, y = c($solution, c($rho_approx)), g = g)
    p2 <- ggplot(data, aes(x, y, group = factor(g), color = factor(g))) +
    geom_line(size = 2) +
    scale_colour_manual(values = c("black", "#F8766D", "#7CAE00", "#00BFC4", "#C77CFF"), labels=glabels) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("mixture_E.eps", p1,  height=5)
    # ggsave("mixture_rho.eps", p2,  height=5)
"""
