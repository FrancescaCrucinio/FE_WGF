push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
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

# pathological example
K(x, y) = 0.595*pdf.(Normal(8.63, 2.56), y .- x) +
        0.405*pdf.(Normal(15.24, 5.39), y .- x);
t = 1:100;
It = ifelse.(t.<=8, exp.(-0.05*(8 .- t).^2), exp.(-0.001*(t .- 8).^2));
It_normalised = ifelse.(t.<=8, exp.(-0.05*(8 .- t).^2), exp.(-0.001*(t .- 8).^2))/31.942;
# renormalise
It = It * 5000/sum(It);
It = round.(It, digits = 0);

# get hospitalisation counts
misspecified = true;
if(misspecified)
    # misspecified
    It_miss = copy(It);
    for i in t[1:98]
        if((mod(i, 6)==0) | (mod(i, 7)==0))
            u = 0.2*rand(1) .+ 0.3;
            proportion = floor.(u[1].*It[i]);
            It_miss[i] = It_miss[i] .- proportion;
            It_miss[i+2] = It_miss[i+2] .+ proportion;
        end
    end
    Isample = vcat(fill.(1:length(It_miss), Int.(It_miss))...);
else
    Isample = vcat(fill.(1:length(It), Int.(It))...);
end
# shuffle sample
shuffle!(Isample);
# well specified
muSample = round.(Isample .+ rand(Gamma(10, 1), length(Isample), 1), digits = 0);
muCounts = counts(Int.(muSample), 100);
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
x0 = sample(muSample, M, replace = false) .- 10;
# prior mean = mean of μ shifted back by 10 days
m0 = mean(muSample) - 10;
sigma0 = std(muSample);
# regularisation parameter
alpha = 0.0008;
runtimeWGF = @elapsed begin
# run WGF
x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M);
end
RKDEyWGF = rks.kde(x = x[end, :], var"eval.points" = t);
KDEyWGF = abs.(rcopy(RKDEyWGF[3]));
# check convergence
EWGF = mapslices(psi, x, dims = 2);
plot(EWGF)

# RL
# initial distribution
pi0 = 100*rand(1, length(muCounts));

KDisc = zeros(length(muCounts), length(muCounts));
for i=1:length(muCounts)
    for j=1:length(muCounts)
        KDisc[i, j] = K(t[j], t[i]);
    end
end
runtimeRL = @elapsed begin
rhoCounts = RL(KDisc, muCounts, 100, pi0);
end
# RIDE estimator
# discretise delay distribution
delay = 0.595*pdf.(Normal(8.63, 2.56), t) +
        0.405*pdf.(Normal(15.24, 5.39), t);
delay = delay./(sum(delay));
R"""
library(tictoc)
library(incidental)
tic()
RIDE_model <- fit_incidence(
  reported = $muCounts,
  delay_dist = $delay)
toc()
RIDE_incidence <- RIDE_model$Ihat
RIDE_reconstruction <- RIDE_model$Chat
"""
# recovolve WGF
refY = t;
delta = refY[2] - refY[1];
KDEyRec = zeros(length(refY), 1);
for i=1:length(refY)
    KDEyRec[i] = delta*sum(K.(t, refY[i]).*KDEyWGF);
end
# recovolve RL
RLyRec = zeros(1, length(refY));
for i=1:length(refY)
    RLyRec[i] = delta*sum(K(t, refY[i]).*rhoCounts[100,:]);
end

p1=plot(t, It)
plot!(p1, t, KDEyWGF*5000)
plot!(p1, t, rhoCounts[100, :])
plot!(p1, t, @rget(RIDE_incidence))

p2=scatter(t, muCounts)
plot!(p2, t, KDEyRec*5000)
plot!(p2, t, RLyRec[:])
plot!(p2, t, @rget(RIDE_reconstruction))

p3=plot(t, It_normalised)
plot!(p3, t, KDEyWGF)
plot!(p3, t, rhoCounts[100, :]/5000)
plot!(p3, t, @rget(RIDE_incidence)/5000)
