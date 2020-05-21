# packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using KernelDensity;
using Interpolations;
using LinearAlgebra;

include("drift_exact_mvn_mean0.jl")

# set seed
Random.seed!(1234);

# variances and means
mu = [0, 0];

sigmaF = [0.1 0; 0 0.1];
sigmaG = [0.45 0.5; 0.5 0.9];
sigmaH = sigmaF + sigmaG;
sigma0 = Matrix{Float64}(I, 2, 2);
rhoG =  sigmaG[1, 2]/sqrt(sigmaG[1, 1] * sigmaG[2, 2]);

ifelse(isposdef(sigmaF) & isposdef(sigmaG), "matrices are positive definite",
        "change covariance matrices!")
# data for anaytically tractable example
f(x) = pdf(MvNormal(mu, sigmaF), x);
h(x) = pdf.(MvNormal(mu, sigmaH), x);
g(x, y) = pdf.(MvNormal(x, sigmaG), y);
fg(y) = pdf(MvNormal(mu, sigma0+sigmaG), y);
# samples from h(y)
M = 100000;
Nparticles = 100000;
hSample = rand(MvNormal(mu, sigmaH), M);
# grid
N = 100;
x = range(-1, 1, length = N);
y = range(-1, 1, length = N);

### approximate drift
x0 = rand(MvNormal(mu, sigma0), Nparticles);
# compute h^N_{n}
hN = zeros(M, 1);
den_exact = zeros(M, 1);
for j=1:M
    hN[j] = mean(pdf(MvNormal(hSample[:, j], sigmaG), x0));
    den_exact[j] = fg(hSample[:, j]);
end
plot(sort!(hN, dims = 1), sort!(den_exact, dims = 1))

driftX = zeros(N, N);
driftY = zeros(N, N);
Threads.@threads for i=1:N
    @simd for j=1:N
        # precompute common quantities for gradient
        prec = pdf(MvNormal([x[i]; y[j]], sigmaG), hSample)/(1 - rhoG^2);
        gradientX = prec .* ((hSample[1, :] .- x[i])/sigmaG[1, 1] -
            rhoG*(hSample[2, :] .- y[j])/sqrt(sigmaG[1, 1]*sigmaG[2, 2]));
        gradientY = prec .* ((hSample[2, :] .- y[j])/sigmaG[2, 2] -
            rhoG*(hSample[1, :] .- x[i])/sqrt(sigmaG[1, 1]*sigmaG[2, 2]));
        driftX[j, i] = mean(gradientX./hN);
        driftY[j, i] = mean(gradientY./hN);
    end
end
p3 = heatmap(x, y, driftX);
title!("N=$Nparticles, M=$M")
p4 = heatmap(x, y, driftY);
