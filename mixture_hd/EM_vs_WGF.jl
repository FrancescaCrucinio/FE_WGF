push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using KernelEstimator;
# custom packages
using wgf_prior;
include("osl_em.jl")

# set seed
Random.seed!(1234);

# mean and variances
means = [0.3; 0.7];
variances = [0.07^2; 0.1^2];

# parameters for penalised KL
# regularisation parameter
alpha = 0.01;
# number of iterations
Niter = 50;
# time discretisation
dt = 1e-2;
# reference measure
m0 = 0.5;
sigma0 = 0.25;
# number of particles
Nparticles = 10^4;
# find bins closest to number of particles
function find_bins(Nparticles, d)
    bins = [[ceil(Nparticles^(1/i))^i for i in 1:d]';
        [floor(Nparticles^(1/i))^i for i in 1:d]'];
    solve = argmin(abs.(bins .- Nparticles), dims = 1);
    optima = [solve[i][1] for i in 1:d];
    Nbins = [bins[optima[i], i]^(1/i) for i in 1:d];
    return Nbins
end
Nbins = trunc.(Int, find_bins(Nparticles, 5));

# 1d marginal
KDEx = range(0, stop = 1, length = 100);
dKDEx = KDEx[2] - KDEx[1];
truth = pdf.(Normal(means[1], sqrt(variances[1])), KDEx)/3 + 2*pdf.(Normal(means[2], sqrt(variances[2])), KDEx)/3;

# number of replicates
Nrep = 2;
tEM = zeros(Nrep, 5);
iseEM = zeros(Nrep, 5);
tWGF = zeros(Nrep, 5);
iseWGF = zeros(Nrep, 5);
for d=1:5
    # mixture of Gaussians
    pi = MixtureModel(MvNormal, [(means[1]*ones(d), diagm(variances[1]*ones(d))), (means[2]*ones(d), diagm(variances[2]*ones(d)))], [1/3, 2/3]);
    sigmaK = 0.15;
    mu = MixtureModel(MvNormal, [(means[1]*ones(d), diagm(variances[1]*ones(d) .+ sigmaK^2)), (means[2]*ones(d), diagm(variances[2]*ones(d) .+ sigmaK^2))], [1/3, 2/3]);

    # discretisation
    # bin centres
    Xbins = range(1/Nbins[d], stop = 1-1/Nbins[d], length = Nbins[d]);
    iter = Iterators.product((Xbins for _ in 1:d)...);
    KDEeval = reduce(hcat, [collect(i) for i in iter])';
    KDEeval = reverse(KDEeval, dims = 2);
    # discretise μ
    muDisc = pdf(mu, KDEeval');
    # reference measure
    pi0 = pdf(MvNormal(m0*ones(d), diagm(sigma0*ones(d))), KDEeval');

    for j=1:Nrep
        # OSL-EM
        tEM[j, d] = @elapsed begin
        resEM, _ = osl_em(muDisc, sigmaK, alpha, Niter, pi0, pi0, KDEeval, false);
        end
        # get closest bin centre for KDEx
        binCENTRE = [searchsortedlast(Xbins, i) for i in KDEx];
        binCENTRE[binCENTRE .== 0] .= 1;
        if(d>1)
            resEMmarginal = [sum(resEM[(i*Nbins[d] + 1):(i+1)*Nbins[d]]) for i in 0:Nbins[d]-1];
            EM = [resEMmarginal[i] for i in binCENTRE];
        else
            # if we have more or equal just choose the closest one
            EM = resEM[binCENTRE];
        end
        iseEM[j, d] = dKDEx * sum((EM .- truth).^2);
        # WGF
        # sample from μ
        muSample = rand(mu, 10^6);
        # initial distribution
        x0 = rand(MvNormal(m0*ones(d), diagm(sigma0*ones(d))), Nparticles);
        tWGF[j, d] = @elapsed begin
        xWGF, _ = wgf_hd_mixture_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, sigmaK, false);
        WGF = kerneldensity(xWGF[1, :], xeval=KDEx);
        end
        iseWGF[j, d] = dKDEx * sum((WGF .- truth).^2);
    end
end
plot(1:5, t_d)
plot(1:5, ise_d)
