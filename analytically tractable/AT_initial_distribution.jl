# push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
using LaTeXStrings;
# custom packages
using diagnostics;
using wgf;

# Compare initial distributions
# pyplot()
# set seed
Random.seed!(1234);

# data for anaytically tractable example
sigmaG = 0.045^2;
sigmaF = 0.043^2;
sigmaH = sigmaF + sigmaG;
f(x) = pdf.(Normal(0.5, sqrt(sigmaF)), x);
h(x) = pdf.(Normal(0.5, sqrt(sigmaH)), x);
g(x, y) = pdf.(Normal(x, sqrt(sigmaG)), y);

# dt and number of iterations
dt = 1e-03;
Niter = 1000;
# samples from h(y)
M = 1000;
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# reference values for KL divergence
refY = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameter
lambda = 0.025;

# initial distributions
x0 = [0.0*ones(1, Nparticles); 0.5*ones(1, Nparticles);
    1*ones(1, Nparticles); rand(1, Nparticles);
    0.5 .+ sqrt(sigmaF)*randn(1, Nparticles);
    0.5 .+ sqrt(sigmaF+0.01)*randn(1, Nparticles)];

m = zeros(Niter-1, size(x0, 1));
v = zeros(Niter-1, size(x0, 1));
q = zeros(Niter-1, size(x0, 1));
misef = zeros(Niter-1, size(x0, 1));
KL = zeros(Niter-1, size(x0, 1));
ent = zeros(Niter-1, size(x0, 1));
E = zeros(Niter-1, size(x0, 1));
# function computing KDE
phi(t) = KernelEstimator.kerneldensity(t, xeval=KDEx, h=bwnormal(t));
# function computing diagnostics
psi(t) = diagnosticsALL(f, h, g, KDEx, t, refY);
# number of repetitions
Nrep = 1;
Threads.@threads for i=1:size(x0, 1)
    mrep = zeros(Niter-1, Nrep);
    vrep = zeros(Niter-1, Nrep);
    qrep = zeros(Niter-1, Nrep);
    misefrep = zeros(Niter-1, Nrep);
    KLrep = zeros(Niter-1, Nrep);
    entrep = zeros(Niter-1, Nrep);
    Erep = zeros(Niter-1, Nrep);
    @simd for k=1:Nrep
        ### WGF
        x, _ = wgf_AT(Nparticles, dt, Niter, lambda, x0[i, :], M);
        # KDE
        KDEyWGF = mapslices(phi, x[2:end, :], dims = 2);
        diagnosticsWGF = mapslices(psi, KDEyWGF, dims = 2);
        # turn into matrix
        diagnosticsWGF = reduce(hcat, getindex.(diagnosticsWGF,j) for j in eachindex(diagnosticsWGF[1]));
        mrep[:, k] = diagnosticsWGF[:, 1];
        vrep[:, k] = diagnosticsWGF[:, 2];
        qrep[:, k] = diagnosticsWGF[:, 3];
        misefrep[:, k] = diagnosticsWGF[:, 4];
        KLrep[:, k] = diagnosticsWGF[:, 5];
        entrep[:, k] = diagnosticsWGF[:, 6];
        Erep[:, k] = diagnosticsWGF[:, 5] .- lambda*diagnosticsWGF[:, 6];
        println("$i, $k")
    end
    m[:, i] = mean(mrep, dims = 2);
    v[:, i] = mean(vrep, dims = 2);
    q[:, i] = mean(qrep, dims = 2);
    misef[:, i] = mean(misefrep, dims = 2);
    KL[:, i] = mean(KLrep, dims = 2);
    ent[:, i] = mean(entrep, dims = 2);
    E[:, i] = mean(Erep, dims = 2);
end
# save data
JLD.save("initial_d1000iter.jld", "x0", x0, "m", m, "v", v, "q", q, "misef", misef,
    "KL", KL, "ent", ent, "E", E);
# load data
# d = load("initial_d.jld");
# m = d["m"];
# v = d["v"];
# q = d["q"];
# misef = d["misef"];
# E = d["E"];
#
# # plot
# labels = [L"$\delta_0$" L"$\delta_{0.5}$" L"$\delta_1$" L"U$[0, 1]$" L"$N(m, \sigma^2_\rho)$" L"$N(m, \sigma^2_\rho+\varepsilon)$"];
# p1 = StatsPlots.plot(2:Niter, m, lw = 3, label = labels, xlabel=L"Iteration",
#     ylabel=L"$\hat{m}_t$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
# p2 = StatsPlots.plot(2:Niter, v, lw = 3, label = labels, xlabel=L"Iteration",
#     ylabel=L"$\hat{v}_t$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
# p3 = StatsPlots.plot(2:Niter, q, lw = 3, label = labels, xlabel=L"Iteration",
#     ylabel=L"$MSE_{95}$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
# p4 = StatsPlots.plot(2:Niter, misef, lw = 3, label = labels, xlabel=L"Iteration",
#     ylabel=L"MISE", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
# p5 = StatsPlots.plot(2:Niter, E, lw = 3, label = labels, xlabel=L"Iteration",
#     ylabel=L"$E(\rho_t)$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
# p6 = StatsPlots.plot();
# plot(p1, p2, p3, p4, p5, p6, layout = (3, 2))
#
# savefig(p2, "initial_distribution_variance.pdf")
# savefig(p3, "initial_distribution_mse.pdf")
# savefig(p4, "initial_distribution_mise.pdf")
# savefig(p5, "initial_distribution_E.pdf")


StatsPlots.plot(10:Niter-1, KL[10:end, :], lw = 3)
