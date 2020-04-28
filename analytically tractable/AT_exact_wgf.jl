push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelDensity;
using Random;
using JLD;
# custom packages
using diagnostics;
using wgf;

# AT: exact minimiser and WGF approximation
# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
h(x) = pdf.(Normal(0.5, sqrt(sigmaH)), x);
g(x, y) = pdf.(Normal(x, sqrt(sigmaG)), y);

# dt and final time
dt = 1e-03;
T = 1;
Niter = trunc(Int, 1/dt);
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
lambda = range(0, stop = 1, length = 3);

# repeat WGF
Nrep = 10;
varWGF = zeros(length(lambda), Nrep);
klWGF = zeros(length(lambda), Nrep);
entWGF = zeros(length(lambda), Nrep);
eWGF = zeros(length(lambda), Nrep);
for i=1:length(lambda)
    for j=1:Nrep
        # initial distribution
        x0 = rand(1, Nparticles);
        ### WGF
        x, _ =  wgf_AT(Nparticles, Niter, lambda[i], x0, M);
        # KDE
        KDEyWGF =  KernelDensity.kde(x[end, :]);
        # evaluate KDE at reference points
        KDEyWGFeval = pdf(KDEyWGF, KDEx);
        KDEyWGFeval[KDEyWGFeval .< 0] .= 0;
        # KL and variance
        _, varWGF[i, j], _, _, klWGF[i, j], entWGF[i, j] = diagnosticsALL(f, h, g, KDEx, KDEyWGFeval, refY);
        eWGF[i, j] = klWGF[i, j] +lambda[i]*entWGF[i, j];
        println("$i, $j")
    end
end

### exact minimiser
varExact, _ = AT_exact_minimiser(sigmaG, sigmaH, lambda);

StatsPlots.plot(lambda, varExact, lw = 3, label = "Exact");
StatsPlots.plot!(lambda, mean(varWGF, dims = 2), lw = 3, label = "WGF");
