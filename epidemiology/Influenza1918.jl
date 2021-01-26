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
@rimport ks as rks;
# custom packages
using wgf_prior;
include("RL.jl")
# set seed
Random.seed!(1234);

# fitted Gaussian approximating K
K(x, y) = 0.595*pdf.(Normal(8.63, 2.56), y .- x) +
        0.405*pdf.(Normal(15.24, 5.39), y .- x);
R"""
library(incidental)
library(tictoc)
# death counts
death_counts <- spanish_flu$Philadelphia

# RIDE estimator
tic()
Philadelphia_model <- fit_incidence(
  reported = spanish_flu$Philadelphia,
  delay_dist = spanish_flu_delay_dist$proportion)
toc()
"""
# get counts from μ
muCounts = Int.(@rget death_counts);
# get sample from μ
muSample = vcat(fill.(1:length(muCounts), muCounts)...);
# shuffle sample
shuffle!(muSample);
# x axis = time (122 days)
KDEx = 1:length(muCounts);
# KDE for μ
RKDE = rks.kde(muSample, var"eval.points" = KDEx);
muKDEy = abs.(rcopy(RKDE[3]));

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
    trueMu = muKDEy;
    refY = KDEx;
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = delta*sum(K.(KDEx, refY[i]).*t);
    end
#    hatMu[iszero.(hatMu)] .= eps();
    kl = kl_divergence(trueMu, hatMu);
    return kl-alpha*ent;
end
# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-3;
# number of iterations
Niter = 5000;
# initial distribution
x0 = sample(muSample, M, replace = true) .- 9;
# prior mean = mean of μ shifted back by 9 days
m0 = 0;
sigma0 = 1;
# regularisation parameter
alpha = 0.016;
runtimeWGF = @elapsed begin
# run WGF
x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M, 0.5);
end
# check convergence
KDEyWGF = mapslices(phi, x, dims = 2);
EWGF = mapslices(psi, KDEyWGF, dims = 2);
p1 = plot(EWGF);
# result
p2 = plot(KDEx, KDEyWGF[Niter, :]);
# deaths distribution
plot!(p2, KDEx .- 9, muKDEy);
p = plot(p1, p2, layout =(2, 1));
p

# RL
# initial distribution
rho0 = [muCounts[9:end]; zeros(8, 1)];
# delay distribution
R"""
K_prop <- spanish_flu_delay_dist$proportion
K_day <- spanish_flu_delay_dist$days
"""
K_prop =@rget K_prop;
K_day = Int.(@rget K_day);

KDisc = eps()*ones(length(muCounts), length(muCounts));
for i=1:length(muCounts)
    for j=1:length(muCounts)
        if (i - j >= 1 && i - j<= length(K_day))
            KDisc[i, j] = K_prop[i - j];
        end
    end
end
runtimeRL = @elapsed begin
rhoCounts = RL(KDisc, muCounts, 100, rho0);
end
# recovolve WGF
refY = KDEx;
delta = refY[2] - refY[1];
KDEyRec = zeros(1, length(refY));
for i=1:length(refY)
    KDEyRec[i] = delta*sum(K.(KDEx, refY[i]).*KDEyWGF[Niter, :]);
end

# recovolve RL
RLyRec = zeros(1, length(refY));
for i=1:length(refY)
    t = refY[i] .- KDEx;
    nonnegative = (t .>= 1) .& (t .<= 31);
        RLyRec[i] = delta*sum(K_prop[t[nonnegative]].*rhoCounts[100, nonnegative]);
end
# plot
R"""
library(ggplot2)
g <- rep(1:3, , each = length(spanish_flu$Date));
data <- data.frame(x = rep(spanish_flu$Date, times = 3), y = c($rhoCounts[100,]/sum($rhoCounts[100,]), Philadelphia_model$Ihat/sum(Philadelphia_model$Ihat), $KDEyWGF[$Niter, ]), g = factor(g))
p1 <- ggplot(data, aes(x, y, color = g)) +
geom_line(size = 1) +
scale_color_manual(values = c("red", "blue", "green"), labels=c("RL", "RIDE", "WGF")) +
theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
# ggsave("flu1918_reconstruction.eps", p1,  height=4)

# reconstructed death counts
g <- rep(1:4, , each = length(spanish_flu$Date));
data <- data.frame(x = rep(spanish_flu$Date, times = 4), y = c($RLyRec, Philadelphia_model$reported, Philadelphia_model$Chat, $KDEyRec*sum(Philadelphia_model$reported)), g = factor(g))
p2 <- ggplot(data, aes(x, y, color = g)) +
geom_point(data = data[data$g==1, ], size = 2, shape = 3) +
geom_line(data = data[data$g!=1, ], size = 1) +
scale_color_manual(values = c("black", "red", "blue", "green"), labels=c("recorded", "RL", "RIDE", "WGF")) +
theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
# ggsave("flu1918_reconv.eps", p2,  height=4)
"""
