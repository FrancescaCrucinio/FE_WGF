push!(LOAD_PATH, "/Users/francescacrucinio/Documents/WGF/myModules")
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
using smcems
# set seed
Random.seed!(1234);

# pathological example
K(x, y) = 0.595*pdf.(Normal(8.63, 2.56), y .- x) +
        0.405*pdf.(Normal(15.24, 5.39), y .- x);
t = 1:100;
It = ifelse.(t.<=8, exp.(-0.05*(8 .- t).^2), exp.(-0.001*(t .- 8).^2));
# renormalise
It = It * 5000/sum(It);
It = round.(It, digits = 0);

# functional approximation
function psiWGF(piSample, alpha, m0, sigma0, muSample)
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
function psiSMC(piSample, W, muSample)
    loglik = zeros(1, length(muSample));
    muN = zeros(M, 1);
    for i=1:length(muSample)
        loglik[i] = sum(W .* K.(piSample, muSample[i]));
    end
    loglik = -log.(loglik);
    kl = mean(loglik);
    return kl;
end
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
# regularisation parameter
alpha = range(0.0001, stop = 0.01, length = 10);
epsilon = range(0, stop = 0.01, length = 10);
# repetitions
L = 5;
EWGF = zeros(length(alpha), L);
ESMC = zeros(length(alpha), L);
for i=1:length(alpha)
    for l=1:L
        # misspecified sample
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
        # initial distribution
        x0 = sample(muSample, M, replace = false) .- 9;
        # prior mean = mean of μ
        m0 = mean(muSample) .- 9;
        sigma0 = std(muSample);
        # WGF
        xWGF = wgf_flu_tamed(Nparticles, dt, Niter_wgf, alpha[i], x0, m0, sigma0, muSample, M);
        # functional
        EWGF[i, l] = psiWGF(xWGF[Niter_wgf, :], alpha[i], m0, sigma0, muSample);
        # SMCEMS
        xSMC, W = smc_flu(Nparticles, Niter_smc, epsilon[i], x0, muSample, M);
        # functional
        ESMC[i, l] = psiSMC(xSMC[Niter_smc, :], W[Niter_smc, :], muSample);
        println("$i, $l")
    end
end
plot(alpha,  mean(EWGF, dims = 2))
plot(epsilon,  mean(ESMC, dims = 2))
