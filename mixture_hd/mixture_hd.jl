push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using LinearAlgebra;
# using OptimalTransport;
using RCall;
@rimport ks as rks;
# custom packages
using smcems;
using wgf_prior;
include("osl_em.jl")

# dimension
p = 2;
# mixture of Gaussians
means = [0.3*ones(1, p); 0.7*ones(1, p)];
variances = [0.07^2; 0.1^2];
pi = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(p))), (means[2, :], diagm(variances[2]*ones(p)))], [1/3, 2/3]);
sigmaK = 0.15;
mu = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(p) .+ sigmaK^2)), (means[2, :], diagm(variances[2]*ones(p) .+ sigmaK^2))], [1/3, 2/3]);

Nbins = 100;
X1bins = range(0, stop = 1, length = Nbins);
X2bins = range(0, stop = 1, length = Nbins);
gridX1 = repeat(X1bins, inner=[Nbins, 1]);
gridX2 = repeat(X2bins, outer=[Nbins 1]);
KDEeval = [gridX1 gridX2];
truth = reshape(pdf(pi, KDEeval'), (Nbins, Nbins));
p1 = heatmap(X1bins, X2bins, truth);


# OSL-EM
Nbins = 50;
X1bins = range(0, stop = 1, length = Nbins);
X2bins = range(0, stop = 1, length = Nbins);
gridX1 = repeat(X1bins, inner=[Nbins, 1]);
gridX2 = repeat(X2bins, outer=[Nbins 1]);
KDEeval = [gridX1 gridX2];
muDisc = pdf(mu, KDEeval');
pi0 = ones(size(KDEeval, 1), 1)/size(KDEeval, 1);
pi_init = ones(size(KDEeval, 1), 1)/size(KDEeval, 1);
alpha = 0.01;
Niter = 100;
tEM = @elapsed begin
EMres = osl_em(muDisc, sigmaK, alpha, Niter, pi0, pi_init);
end
EM = reshape(EMres, (Nbins, Nbins));
p2 = heatmap(X1bins, X2bins, EM);

# SMC-EMS
muSample = rand(mu, 10^6);
x0 = rand(p, Nbins^p);
epsilon = 1e-03;
tSMC = @elapsed begin
xSMC, W = smc_p_dim_gaussian_mixture(Nbins^p, Niter, epsilon, x0, muSample, sigmaK);
# KDE
bw1 = sqrt(epsilon^2 + optimal_bandwidthESS(x[1, :], W)^2);
bw2 = sqrt(epsilon^2 + optimal_bandwidthESS(x[2, :], W)^2);
Rkde = rks.kde(x = [x[1, :] x[2, :]], var"eval.points" = KDEeval, var"w" = Nbins^p*W, var"H" = [bw1^2 0; 0 bw2^2]);
SMCkde = abs.(rcopy(Rkde[3]));
end
SMC_EMS = reshape(SMCkde, (Nbins, Nbins));
p3 = heatmap(X1bins, X2bins, SMC_EMS);

# WGF
m0 = 0.5;
sigma0 = 0.1;
dt = 1e-3;
tWGF = @elapsed begin
xWGF = wgf_hd_mixture_tamed(Nbins^p, dt, Niter, alpha, x0, m0, sigma0, muSample, sigmaK);
Rkde = rks.kde(x = [xWGF[1, :] xWGF[2, :]], var"eval.points" = KDEeval);
WGFkde = abs.(rcopy(Rkde[3]));
end
WGF = reshape(WGFkde, (Nbins, Nbins));
p4 = heatmap(X1bins, X2bins, WGF);

plot(p1, p2, p3, p4, layout = (2,2))
#
piSample = rand(pi, size(KDEeval, 1));
piWeights = fill(1/size(KDEeval, 1), size(KDEeval, 1));

C = pairwise(Cityblock(), piSample, KDEeval', dims = 2);
OptimalTransport.emd2(piWeights, EMres[:]./sum(EMres), C)

C = pairwise(Cityblock(), piSample, xSMC, dims = 2);
OptimalTransport.emd2(piWeights, W, C)

C = pairwise(Cityblock(), piSample, xWGF, dims = 2);
OptimalTransport.emd2(piWeights, fill(1/Nbins^p, Nbins^p), C)
