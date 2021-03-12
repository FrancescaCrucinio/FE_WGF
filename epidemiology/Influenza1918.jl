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
Sys.setenv("LANGUAGE"="En")
Sys.setlocale("LC_ALL", "English")

# death counts
death_counts <- spanish_flu$Philadelphia

# RIDE estimator
tic()
Philadelphia_model <- fit_incidence(
  reported = spanish_flu$Philadelphia,
  delay_dist = spanish_flu_delay_dist$proportion)
toc()
RIDE_reconstruction <- Philadelphia_model$Chat/sum(Philadelphia_model$Chat)
RIDE_incidence <- Philadelphia_model$Ihat
"""
# get counts from μ
muCounts = Int.(@rget death_counts);
# get sample from μ
muSample = vcat(fill.(1:length(muCounts), muCounts)...);
# shuffle sample
shuffle!(muSample);
# x axis = time (122 days)
KDEx = 1:length(muCounts);

# functional approximation
function psi(piSample)
    loglik = zeros(1, length(muSample));
    for i=1:length(muSample)
        loglik[i] = mean(K.(piSample, muSample[i]));
    end
    loglik = -log.(loglik);
    kl = mean(loglik);
    prior = pdf.(Normal(m0, sigma0), piSample);
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    kl_prior = mean(log.(pihat./prior));
    return kl+alpha*kl_prior;
end

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-1;
# number of iterations
Niter = 3000;
# initial distribution
x0 = sample(muSample, M, replace = false) .- 9;
# prior mean = mean of μ shifted back by 9 days
m0 = mean(muSample) - 9;
sigma0 = std(muSample);
# regularisation parameter
alpha = 0.0002;
# run WGF
runtimeWGF = @elapsed begin
x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M);
RKDEyWGF = rks.kde(x = x[Niter, :], var"eval.points" = KDEx);
KDEyWGF = abs.(rcopy(RKDEyWGF[3]));
end
# check convergence
EWGF = mapslices(psi, x, dims = 2);
plot(EWGF)

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
rhoCounts = RL(KDisc, muCounts, 200, rho0);
end
# recovolve WGF
refY = KDEx;
delta = refY[2] - refY[1];
KDEyRec = zeros(length(refY), 1);
for i=1:length(refY)
    KDEyRec[i] = delta*sum(K.(KDEx, refY[i]).*KDEyWGF);
end
KDEyRec = KDEyRec/sum(KDEyRec);

# recovolve RL
RLyRec = zeros(length(refY), 1);
for i=1:length(refY)
    t = refY[i] .- KDEx;
    nonnegative = (t .>= 1) .& (t .<= 31);
        RLyRec[i] = delta*sum(K_prop[t[nonnegative]].*rhoCounts[200, nonnegative]);
end
RLyRec = RLyRec/sum(RLyRec);

estimators = [rhoCounts[200, :]/sum(rhoCounts[200, :]) @rget(RIDE_incidence)/sum(@rget(RIDE_incidence)) KDEyWGF];
p1=plot(KDEx, estimators, lw = 1, label = ["RL" "RIDE" "WGF"],
    color = [:gray :blue :red], line=[:solid :solid :solid],
    legendfontsize = 15, tickfontsize = 10)
# savefig(p1,"1918flu_incidence.pdf")

reconvolutions = [RLyRec[:] @rget(RIDE_reconstruction) KDEyRec].*sum(muCounts);
p2=scatter(KDEx, muCounts, marker=:x, markersize=3, label = "reported cases", color = :black)
plot!(p2, KDEx, reconvolutions, lw = 1, label = ["RL" "RIDE" "WGF"],
    color = [:gray :blue :red], line=[:solid :solid :solid],
    legendfontsize = 15, tickfontsize = 10)
# savefig(p2,"1918flu_reconvolution.pdf")
