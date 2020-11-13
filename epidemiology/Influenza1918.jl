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
using RCall;
@rimport ks as rks;
# custom packages
using wgf;

# set seed
Random.seed!(1234);

R"""
library(incidental)
# death counts
death_counts <- spanish_flu$Philadelphia
# fit Gamma to delay distribution
x <- sample(spanish_flu_delay_dist$days, 1000000, replace = TRUE, prob = spanish_flu_delay_dist$proportion)
fit.gamma <- MASS::fitdistr(x, "gamma")
gamma_shape <- fit.gamma$estimate[1]
gamma_rate <- 1/fit.gamma$estimate[2]
"""
# get Gamma parameters
gamma_shape = @rget gamma_shape;
gamma_scale =  @rget gamma_rate;
# get counts from μ
muCounts = Int.(@rget death_counts);
# get sample from μ
muSample = vcat(fill.(1:length(muCounts), muCounts)...);
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
        hatMu[i] = delta*sum(pdf.(Gamma.(gamma_shape, gamma_rate), refY[i] .- KDEx).*t);
    end
    kl = kl_divergence(trueMu, hatMu);
    return kl-alpha*ent;
end

# parameters for WGF
# number of particles
Nparticles = 1000;
# number of samples from μ to draw at each iteration
M = 1000;
# time discretisation
dt = 1e-3;
# number of iterations
Niter = 5000;
# initial distribution
x0 = sample(muSample, M, replace = true) .- 9;
# regularisation parameter
alpha = 0.001;
# run WGF
x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, 0.5, gamma_shape, gamma_scale);

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

# recovolve
muSampleReconstructed = x[Niter, :] +  rand(Gamma(gamma_shape, gamma_rate), Nparticles);
RKDE = rks.kde(muSampleReconstructed, var"eval.points" = KDEx);
KDEyRec = abs.(rcopy(RKDE[3]));
R"""
Philadelphia_model <- fit_incidence(
  reported = spanish_flu$Philadelphia,
  delay_dist = spanish_flu_delay_dist$proportion)
plot(spanish_flu$Date, Philadelphia_model$Ihat/sum(Philadelphia_model$Ihat), type="l", col = "red")
lines(spanish_flu$Date, $KDEyWGF[$Niter, ], col = "blue")
lines(spanish_flu$Date, $KDEyRec, col = "cyan")
lines(spanish_flu$Date, Philadelphia_model$Chat/sum(Philadelphia_model$Chat), col = "orange")
points(spanish_flu$Date, Philadelphia_model$reported/sum(Philadelphia_model$reported), pch = 16, col = "black")
"""
