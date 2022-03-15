push!(LOAD_PATH, "/Users/francescacrucinio/Documents/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
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
using smcems;
include("RL.jl")
R"""
library(tictoc)
library(incidental)
"""
# set seed
Random.seed!(1234);

# synthetic data
K(x, y) = 0.595*pdf.(Normal(8.63, 2.56), y .- x) +
        0.405*pdf.(Normal(15.24, 5.39), y .- x);
t = 1:100;
It = ifelse.(t.<=8, exp.(-0.05*(8 .- t).^2), exp.(-0.001*(t .- 8).^2))/31.942;
It_normalised = copy(It);
# renormalise
It = It * 5000/sum(It);
It = round.(It, digits = 0);

# discretise K for RL
KDisc = zeros(length(It), length(It));
for i=1:length(It)
    for j=1:length(It)
        KDisc[i, j] = K(t[j], t[i]);
    end
end

# discretise K for RIDE
delay = 0.595*pdf.(Normal(8.63, 2.56), t) +
        0.405*pdf.(Normal(15.24, 5.39), t);
delay = delay./(sum(delay));

refY = t;
delta = refY[2] - refY[1];
# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-1;
# number of iterations
Niter_wgf = 3000;
Niter_smc = 100;
Niter_rl = 100;
# regularisation parameter
alpha = 0.001;
epsilon = 0.0002;

# misspecified or not
misspecified = false;
Nrep = 100;
ise = zeros(4, Nrep);
ise_reconvolved = zeros(4, Nrep);
runtime = zeros(4, Nrep);
for i=1:Nrep
    if(misspecified)
        # misspecified model
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
        # shuffle sample
        shuffle!(Isample);
        # well specified
        muSample = round.(Isample .+ rand(MixtureModel(Normal, [(8.63, 2.56), (15.24, 5.39)], [0.595, 0.405]), length(Isample), 1), digits = 0);
        muCounts = counts(Int.(muSample), 100);
    else
        Isample = vcat(fill.(1:length(It), Int.(It))...);
        # shuffle sample
        shuffle!(Isample);
        # well specified
        muSample = round.(Isample .+ rand(MixtureModel(Normal, [(8.63, 2.56), (15.24, 5.39)], [0.595, 0.405]), length(Isample), 1), digits = 0);
        muCounts = counts(Int.(muSample), 100);
    end

    # RL
    # initial distribution
    pi0 = 100*rand(1, length(muCounts));
    runtime[1, i] = @elapsed begin
    rhoCounts = RL(KDisc, muCounts, Niter_rl, pi0);
    end
    ise[1, i] = sum((rhoCounts[Niter_rl, :]/5000 .- It_normalised).^2);
    # recovolve RL
    RLyRec = zeros(length(refY), 1);
    for i=1:length(refY)
        RLyRec[i] = delta*sum(K(t, refY[i]).*rhoCounts[Niter_rl,:]);
    end
    RLyRec = RLyRec/sum(RLyRec);
    ise_reconvolved[1, i] = sum((RLyRec .- muCounts/5000).^2);

    # RIDE
    R"""
    tic()
    RIDE_model <- fit_incidence(
      reported = $muCounts,
      delay_dist = $delay)
    exectime <- toc()
    RIDE_exectime <- exectime$toc - exectime$tic
    RIDE_incidence <- RIDE_model$Ihat
    # reconvolve RIDE
    RIDE_reconstruction <- RIDE_model$Chat/sum(RIDE_model$Chat)
    """
    runtime[2, i] = @rget RIDE_exectime;
    ise[2, i] = sum((@rget(RIDE_incidence)/5000 .- It_normalised).^2);
    ise_reconvolved[2, i] = sum((@rget(RIDE_reconstruction) .- muCounts/5000).^2);

    # WGF
    # initial distribution
    x0 = sample(muSample, M, replace = false) .- 9;
    # prior mean = mean of μ shifted back by 10 days
    m0 = mean(muSample) - 9;
    sigma0 = std(muSample);
    runtime[3, i] = @elapsed begin
    xWGF = wgf_flu_tamed_truncated(Nparticles, dt, Niter_wgf, alpha, x0, m0, sigma0, muSample, M);
    RKDEyWGF = rks.kde(x = xWGF[Niter_wgf, :], var"eval.points" = t);
    KDEyWGF = abs.(rcopy(RKDEyWGF[3]));
    end
    ise[3, i] = sum((KDEyWGF .- It_normalised).^2);
    # recovolve WGF
    KDEyRec = zeros(length(refY), 1);
    for i=1:length(refY)
        KDEyRec[i] = delta*sum(K.(t, refY[i]).*KDEyWGF);
    end
    KDEyRec = KDEyRec/sum(KDEyRec);
    ise_reconvolved[3, i] = sum((KDEyRec .- muCounts/5000).^2);
    # SMCEMS
    runtime[4, i] = @elapsed begin
    xSMC, W = smc_flu(Nparticles, Niter_smc, epsilon, x0, muSample, M);
    bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter_smc, :], W[Niter_smc, :])^2);
    RKDESMC = rks.kde(x = xSMC[Niter_smc,:], var"h" = bw, var"eval.points" = t, var"w" = Nparticles*W[Niter_smc, :]);
    KDEySMC =  abs.(rcopy(RKDESMC[3]));
    end
    ise[4, i] = sum((KDEySMC .- It_normalised).^2);
    # recovolve SMCEMS
    KDEyRecSMCEMS = zeros(length(refY), 1);
    for i=1:length(refY)
        KDEyRecSMCEMS[i] = delta*sum(K.(t, refY[i]).*KDEySMC);
    end
    KDEyRecSMCEMS = KDEyRecSMCEMS/sum(KDEyRecSMCEMS);
    ise_reconvolved[4, i] = sum((KDEyRecSMCEMS .- muCounts/5000).^2);
end
mean(ise, dims = 2)
mean(ise_reconvolved, dims = 2)
times = mean(runtime, dims = 2)
using JLD;
save("sim_epidem10Mar2022_truncated.jld", "runtime", runtime, "ise", ise, "ise_reconvolved", ise_reconvolved);
# ise = load("sim_epidem10Mar2022misspecified.jld", "ise");
# runtime = load("sim_epidem10Mar2022misspecified.jld", "runtime");
# ise_reconvolved = load("sim_epidem10Mar2022misspecified.jld", "ise_reconvolved");
