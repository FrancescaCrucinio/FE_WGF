push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using LinearAlgebra;
include("mixture_hd/osl_em.jl")

# dimension
p = 2;
# mixture of Gaussians
means = [0.3*ones(1, p); 0.7*ones(1, p)];
variances = [0.07^2; 0.1^2];
pi = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(p))), (means[2, :], diagm(variances[2]*ones(p)))], [1/3, 2/3]);
sigmaK = 0.15;
mu = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(p) .+ sigmaK^2)), (means[2, :], diagm(variances[2]*ones(p) .+ sigmaK^2))], [1/3, 2/3]);

X1bins = range(0, stop = 1, length = 11);
X2bins = range(0, stop = 1, length = 11);
gridX1 = repeat(X1bins, inner=[11, 1]);
gridX2 = repeat(X2bins, outer=[11 1]);
KDEeval = [gridX1 gridX2];


muDisc = pdf(mu, KDEeval');
pi0 = ones(size(KDEeval, 1), 1)/size(KDEeval, 1);
pi_init = ones(size(KDEeval, 1), 1)/size(KDEeval, 1);
alpha = 0.01;
Niter = 100;
EMSres = osl_em(muDisc, sigmaK, alpha, Niter, pi0, pi_init);
EMS = reshape(res, (11, 11));
heatmap(X1bins, X2bins, EMS)

sum(EMSres.*KDEeval[:, 1])/sum(EMSres)
