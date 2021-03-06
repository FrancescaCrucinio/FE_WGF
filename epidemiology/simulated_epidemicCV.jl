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
function psi(piSample, alpha, m0, sigma0)
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
alpha = range(0.0001, stop = 0.005, length = 10);

# repetitions
L = 5;
E = zeros(length(alpha), L);
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
        muSample = round.(Isample .+ rand(Gamma(10, 1), length(Isample), 1), digits = 0);
        # initial distribution
        x0 = sample(muSample, M, replace = false) .- 10;
        # prior mean = mean of μ
        m0 = mean(muSample) .- 10;
        sigma0 = std(muSample);
        # WGF
        x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M);
        # functional
        E[i, l] = psi(x[Niter, :], alpha[i], m0, sigma0);
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
