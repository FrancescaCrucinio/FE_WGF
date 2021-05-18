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
using OptimalTransport;
using RCall;
@rimport ks as rks;
# custom packages
using wgf_prior;
include("osl_em.jl")

# set seed
Random.seed!(1234);

# dimension
d = 2;
# mixture of Gaussians
means = [0.3*ones(1, d); 0.7*ones(1, d)];
variances = [0.07^2; 0.1^2];
pi = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(d))), (means[2, :], diagm(variances[2]*ones(d)))], [1/3, 2/3]);
sigmaK = 0.15;
mu = MixtureModel(MvNormal, [(means[1, :], diagm(variances[1]*ones(d) .+ sigmaK^2)), (means[2, :], diagm(variances[2]*ones(d) .+ sigmaK^2))], [1/3, 2/3]);

# parameters for penalised KL
alpha = 0.01;
Niter = 100;
m0 = 0.5;
sigma0 = 0.25;
Nparticles = 10^2;
Nbins = trunc(Integer, ceil(Nparticles^(1/d)));

# OSL-EM
Xbins = range(0, stop = 1, length = Nbins);
iter = Iterators.product((Xbins for _ in 1:d)...);
KDEeval = reduce(vcat, vec([collect(i) for i in iter])');
# discretise μ
muDisc = pdf(mu, KDEeval');
# reference measure
pi0 = pdf(MvNormal(m0*ones(d), diagm(sigma0*ones(d))), KDEeval');
tEM = @elapsed begin
EMres, funEM = osl_em(muDisc, sigmaK, alpha, Niter, pi0, pi0, KDEeval);
end
plot(funEM)
# WGF
# time discretisation
dt = 1e-2;
# sample from μ
muSample = rand(mu, 10^6);
# initial distribution
x0 = rand(MvNormal(m0*ones(d), diagm(sigma0*ones(d))), Nparticles);
tWGF = @elapsed begin
xWGF, funWGF = wgf_hd_mixture_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, sigmaK);
#Rkde = rks.kde(x = [xWGF[1, :] xWGF[2, :]], var"eval.points" = KDEeval);
#WGFres = abs.(rcopy(Rkde[3]));
end
plot(funWGF)
