# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;

sigmaK = 0.045^2;
sigmaRho = 0.043^2;
sigmaMu = sigmaRho + sigmaK;

beta = range(0, stop = 0.2, length = 100);
alpha = 0.5;

f(beta, alpha) = (2pi*beta).^(alpha)*exp(0.5*(alpha + 1 .- sigmaMu./(beta .+ sigmaK))) *
    ((beta .+ sigmaK)./sigmaMu)^(-0.5);
ent(beta) = 0.5*(1+log.(2pi*beta));
fBeta = f.(beta, alpha);
plot(beta, fBeta)

# SGD
alpha = zeros(1, 100000);
alpha[1] = 0.01;
gamma = 1e-04;
for i=2:100000
    fBeta = f.(beta, alpha[i-1]);
    betaSample = sample(beta, Weights(fBeta), 1000, replace = true);
    alpha[i] = alpha[i-1] .- gamma*mean(ent.(betaSample));
end
plot(alpha[:])
