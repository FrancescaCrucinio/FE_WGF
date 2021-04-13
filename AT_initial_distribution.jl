push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using StatsPlots;
using KernelEstimator;
using Random;
using JLD;
using Distances;
using RCall;
@rimport ks as rks
# custom packages
using wgf_prior;

# Compare initial distributions

# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaK = 0.45^2;
sigmaPi = 0.43^2;
sigmaMu = sigmaPi + sigmaK;
pi(x) = pdf.(Normal(0, sqrt(sigmaPi)), x);
mu(x) = pdf.(Normal(0, sqrt(sigmaMu)), x);
K(x, y) = pdf.(Normal(x, sqrt(sigmaK)), y);

# dt and number of iterations
dt = 1e-2;
Niter = 300;
# samples from h(y)
M = 500;
# values at which evaluate KDE
KDEx = range(-2, stop = 2, length = 1000);
# number of particles
Nparticles = 500;
# regularisation parameter
alpha = 0.02;

# initial distributions
x0 = sigmaPi * randn(1, Nparticles);
# x0 = 4*rand(1, Nparticles) .- 2;
# x0 = 0.0001 * randn(1, Nparticles);
# sample from μ
muSample = rand(Normal(0, sqrt(sigmaMu)), 10^4);
m0 = mean(muSample);
sigma0 = std(muSample);
# size of sample from μ
M = min(Nparticles, length(muSample));

reference = ["uniform" "normal" "t" "Laplace"];
E = zeros(Niter, length(reference));
KDEy = zeros(length(KDEx), length(reference));
for i=1:length(reference)
    x, E[:, i] = wgf_AT_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M, reference[i]);
    RKDEyWGF = rks.kde(x = x[Niter, :], var"eval.points" = KDEx);
    KDEy[:, i] = abs.(rcopy(RKDEyWGF[3]));
end

p1 = plot(KDEx, pi(KDEx), color = :black, label = "pi", lw = 4)
# plot!(p1, KDEx, mu(KDEx), color = :gray, label = "mu", lw = 2)
plot!(p1, KDEx, KDEy, label = ["uniform" "normal" "t" "Laplace"], lw = 3, line = :dash, color = [1 2 3 4], legendfontsize = 15, tickfontsize = 10)
# savefig(p1, "pi0_pi_pi.pdf")

p2 = plot(KDEx, ones(length(KDEx), 1))
plot!(p2, KDEx, pdf.(Normal(0, sigma0), KDEx))
plot!(p2, KDEx, pdf.(TDist(100), KDEx))
plot!(p2, KDEx, pdf.(Laplace(0, sigma0/sqrt(2)), KDEx))

p3 = plot(log.(E), label = ["uniform" "normal" "t" "Laplace"], lw = 3, legendfontsize = 15, tickfontsize = 10)
# savefig(p3, "pi0_pi_E.pdf")


# approximate KL
sample = sigmaPi * randn(1, 10^6);
gaussian_ref = mean(pdf.(Normal(0, sigma0), sample));
t_ref = mean(pdf.(TDist(100), sample));
laplace_ref = mean(pdf.(Laplace(0, sigma0/sqrt(2)), sample));
