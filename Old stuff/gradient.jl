using Distributions;
using Plots;
using Random;
include("drift_exact.jl")
include("drift_approximate.jl")

# set seed
Random.seed!(1234);

x = range(0, 1, length = 1000);
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
sigma0 = 0.01;

g(x, y) = (y .- x)./sigmaG .* pdf.(Normal.(x, sqrt(sigmaG)), y);
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);

driftE = drift_exact(0.5, sigma0, sigmaG, sigmaH, x);
p3 = plot(x, driftE)
title!("exact drift")

N = [100 500 1000 5000 10000];
M = 1000;
driftA = zeros(length(x), length(N));
for i=1:length(N)
    driftA[:, i] = drift_approximate(0.5, sigma0, sigmaG, sigmaH, x, M, N[i]);
end
p4 = plot(x, driftA)
title!("approximate drift")
plot(p3, p4, layout=(2, 1))
