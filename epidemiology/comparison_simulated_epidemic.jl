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
R"""
library(tictoc)
library(incidental)
"""
# set seed
Random.seed!(1234);

# pathological example
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

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-1;
# number of iterations
Niter = 3000;
# regularisation parameter
alpha = 0.0009;

# misspecified or not
misspecified = true;
Nrep = 100;
ise = zeros(3, Nrep);
ise_reconvolved = zeros(3, Nrep);
runtime = zeros(3, Nrep);
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
    rhoCounts = RL(KDisc, muCounts, 100, pi0);
    end
    ise[1, i] = sum((rhoCounts[100, :]/5000 .- It_normalised).^2);
    # recovolve RL
    RLyRec = zeros(length(refY), 1);
    for i=1:length(refY)
        RLyRec[i] = delta*sum(K(t, refY[i]).*rhoCounts[200,:]);
    end
    ise_reconvolved[1, i] = sum((RLyRec/5000 .- muCounts/5000).^2);

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
    RIDE_reconstruction <- RIDE_model$Chat
    """
    runtime[2, i] = @rget RIDE_exectime;
    ise[2, i] = sum((@rget(RIDE_incidence)/5000 .- It_normalised).^2);
    ise_reconvolved[2, i] = sum((@rget(RIDE_reconstruction)/5000 .- muCounts/5000).^2);

    # WGF
    # initial distribution
    x0 = sample(muSample, M, replace = false) .- 10;
    # prior mean = mean of μ shifted back by 10 days
    m0 = mean(muSample) - 10;
    sigma0 = std(muSample);
    runtime[3, i] = @elapsed begin
    x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M);
    RKDEyWGF = rks.kde(x = x[Niter, :], var"eval.points" = t);
    KDEyWGF = abs.(rcopy(RKDEyWGF[3]));
    end
    ise[3, i] = sum((KDEyWGF .- It_normalised).^2);
    # recovolve WGF
    refY = t;
    delta = refY[2] - refY[1];
    KDEyRec = zeros(length(refY), 1);
    for i=1:length(refY)
        KDEyRec[i] = delta*sum(K.(t, refY[i]).*KDEyWGF);
    end
    ise_reconvolved[3, i] = sum((KDEyRec/5000 .- muCounts/5000).^2);
end
mean(ise, dims = 2)
times = mean(runtime, dims = 2);
using JLD;
save("sim_epidem10Mar2021misspecified.jld", "runtime", runtime, "ise", ise, "ise_reconvolved", ise_reconvolved);
# ise = load("sim_epidem9Mar2021misspecified.jld", "ise");
# runtime = load("sim_epidem9Mar2021misspecified.jld", "runtime");
