push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/Package")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelDensity;
# custom modules
using Fredholm_WGF;

sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
KDEx = range(0, stop = 1, length = 1000);

h(M) = rand(Normal(0.5, sqrt(sigmaF + sigmaG)), M);
gradient(x, y) = pdf(Normal(x, sqrt(sigmaG)), y) * (y - x)/sigmaG;
g(x, y) = pdf(Normal(x, sqrt(sigmaG)), y);

fie=FIE(h, g, [0 1]);

x= wgf_solve(fie, 0.025, gradient, dt=1e-02);

KDEyWGF =  KernelDensity.kde(x[end, :]);
# evaluate KDE at reference points
KDEyWGFeval = pdf(KDEyWGF, KDEx);

# plot
p = StatsPlots.plot(f, 0, 1, lw = 3, label = "True f")
StatsPlots.plot!(KDEx, KDEyWGFeval, lw = 3, label = "WGF")
