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

sigmaF = [0.15 -0.0645; -0.0645 0.43];
sigmaG = [0.45 0.5; 0.5 0.9];
sigmaH = sigmaF + sigmaG;
sigma0 = Matrix{Float64}(I, 2, 2);

ifelse(isposdef(sigmaF) & isposdef(sigmaG), "matrices are positive definite",
        "change covariance matrices!")
# data for anaytically tractable example
f(x) = pdf(MvNormal(mu, sigmaF), x);
h(x) = pdf.(MvNormal(mu, sigmaH), x);
g(x, y) = pdf.(MvNormal(x, sigmaG), y);

# samples from h(y)
M = 100000;
hSample = rand(MvNormal(mu, sigmaH), M);
# grid
N = 100;
x = range(-2, 2, length = N);
y = range(-2, 2, length = N);

### approximate drift
x0 = rand(MvNormal(mu, sqrt(sigma0)), 1000);
# compute h^N_{n}
hN = zeros(M, 1);
for j=1:M
    # define Gaussian pdf
    phi(t) = pdf(MvNormal(hSample[:, j], sigmaG), t);
    # apply it to c, y
    hN[j] = mean(mapslices(phi, x0, dims = 1));
end
# gradient and drift
driftX = zeros(N, N);
driftY = zeros(N, N);
for i=1:N
    for j=1:N
        # precompute common quantities for gradient
        # define Gaussian pdf
        psi(t) = pdf(MvNormal([x[i]; y[j]], sigmaG), t);
        prec = mapslices(psi, hSample', dims = 2) .*
            ((hSample[2, :] .- y[j])/sqrt(sigmaG[2, 2]) -
            (hSample[1, :] .- x[i])/sqrt(sigmaG[1, 1]));
        prec = sigmaG[2, 2] * sigmaG[1, 1] *
            prec./(sigmaG[2, 2] * sigmaG[1, 1] - sigmaG[1, 2]^2);
        gradientX = -prec./sqrt(sigmaG[1, 1]);
        gradientY = prec./sqrt(sigmaG[2, 2]);
        driftX[N-i+1, j] = mean(gradientX./hN);
        driftY[N-i+1, j] = mean(gradientY./hN);
    end
end

p1 = heatmap(x, y, driftX);
p2 = heatmap(x, y, driftY);
plot(p1, p2, layout = (2, 1))
title!("M = $M")

# fplot = zeros(N, N);
# for i=1:N
#     for j=1:N
#         fplot[N-i+1, j] = f([x[i]; y[j]]);
#     end
# end
# heatmap(x, y, fplot)
### exact drift
drift1, drift2 = drift_exact_mvn_mean0(sigma0, sigmaG, sigmaH, x, y);
p3 = heatmap(x, y, drift1);
p4 = heatmap(x, y, drift2);
plot(p3, p4, layout = (2, 1))
