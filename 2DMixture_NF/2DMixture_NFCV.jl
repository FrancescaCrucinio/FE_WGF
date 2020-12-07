push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
# R
using RCall;
@rimport ks as rks;
# custom packages
using wgf;

# set seed
Random.seed!(1234);

# function computing KDE
# grid
X1bins = range(-5, stop = 5, length = 50);
X2bins = range(-5, stop = 5, length = 50);
gridX1 = repeat(X1bins, inner=[50, 1]);
gridX2 = repeat(X2bins, outer=[50 1]);
KDEeval = [gridX1 gridX2];
function phi(t)
    RKDE = rks.kde(x = [t[1:Nparticles] t[(Nparticles+1):(2Nparticles)]], var"eval.points" = KDEeval);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
a = 1;
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueMu = pdf(mu, collect(transpose(KDEeval)));
    refY1 = X1bins;
    refY2 = X2bins;
    # approximated value
    delta1 = refY1[2] - refY1[1];
    delta2 = refY2[2] - refY2[1];
    hatMu = zeros(length(refY2), length(refY1));
    # convolution with approximated ρ
    # this gives the approximated value
    for i=1:length(refY2)
        for j=1:length(refY1)
            hatMu[i, j] = delta1*delta2*sum(pdf(MvNormal([refY1[j]; refY2[i]], sigmaK), transpose(KDEeval)).*t);
        end
    end
    kl = kl_divergence(trueMu[:], hatMu[:]);
    return kl-a*ent;
end

# data for NF mixture example
sigmaK = [0.1 0; 0 1];
rho = MixtureModel(MvNormal, [([-2, 0], [0.3^2 0; 0 1]), ([0, -2], [1 0; 0 0.3^2]), ([0, 2], [1 0; 0 0.3^2])]);
mu = MixtureModel(MvNormal, [([-2, 0], [0.3^2 0; 0 1] .+ sigmaK), ([0, -2], [1 0; 0 0.3^2] .+ sigmaK), ([0, 2], [1 0; 0 0.3^2] .+ sigmaK)]);
# dt and number of iterations
dt = 1e-01;
Niter = 500;
# samples from μ(y)
M = 1000;
# number of particles
Nparticles = 1000;
# regularisation parameters
alpha = range(0.0005, stop = 0.0011, length = 20);

x0 = rand(mu, Nparticles);

L = 20;
muSample = rand(mu, 100000);

E = zeros(length(alpha), L);
for i=1:length(alpha)
    for l=1:L
        # get reduced sample
        muSampleL = muSample[:, setdiff(1:100000, Tuple(((1:5000) .+ (l-1)*5000)))];
        # WGF
        x1, x2 = wgf_mvnormal_tamed(Nparticles, dt, Niter, alpha[i], x0, muSampleL, M, 0.5);
        # KL
        a = alpha[i];
        KDE = phi([x1[Niter, :]; x2[Niter, :]]);
        E[i, l] = psi(KDE);
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
