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
# custom packages
using diagnostics;
using wgf;

# Compare initial distributions

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
lambda = 0.025;

# initial distributions
x0 = [0.0*ones(1, Nparticles); 0.5*ones(1, Nparticles);
    1*ones(1, Nparticles); rand(1, Nparticles);
    0.5 .+ sqrt(sigmaF)*randn(1, Nparticles)];

E = zeros(Niter-1, size(x0, 1));
m = zeros(Niter-1, size(x0, 1));
v = zeros(Niter-1, size(x0, 1));
q = zeros(Niter-1, size(x0, 1));
misef = zeros(Niter-1, size(x0, 1));

# function computing KDE
phi(t) = KernelEstimator.kerneldensity(t, xeval=KDEx, h=bwnormal(t));
# function computing diagnostics
psi(t) = diagnosticsALL(f, h, g, KDEx, t, refY);
Threads.@threads for i=1:size(x0, 1)
    ### WGF
    x, _ = wgf_AT(Nparticles, Niter, lambda, x0[i, :], M);
    # KDE
    KDEyWGF = mapslices(phi, x[2:end, :], dims = 2);
    diagnosticsWGF = mapslices(psi, KDEyWGF, dims = 2);
    # turn into matrix
    diagnosticsWGF = reduce(hcat, getindex.(diagnosticsWGF,j) for j in eachindex(diagnosticsWGF[1]));
    m[:, i] = diagnosticsWGF[:, 1];
    v[:, i] = diagnosticsWGF[:, 2];
    q[:, i] = diagnosticsWGF[:, 3];
    misef[:, i] = diagnosticsWGF[:, 4];
    E[:, i] = diagnosticsWGF[:, 5]-lambda*diagnosticsWGF[:, 6];
    println("$i finished")
end

# load data
d = load("initial_d.jld");
m = d["m"];
v = d["v"];
q = d["q"];
misef = d["misef"];
E = d["E"];

# plot
times = range(0, stop = 1, length = Niter);
labels = [L"$\delta_0$" L"$\delta_{0.5}$" L"$\delta_1$" L"U$[0, 1]$" L"$N(m, \sigma^2_\rho)$" L"$N(m, \sigma^2_\rho+\varepsilon)$"];
p1 = StatsPlots.plot(times[2:end], m, lw = 3, label = labels, xlabel=L"$t$",
    ylabel=L"$\hat{m}_t$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p2 = StatsPlots.plot(times[2:end], v, lw = 3, label = labels, xlabel=L"$t$",
    ylabel=L"$\hat{v}_t$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p3 = StatsPlots.plot(times[101:end], q[100:end, :], lw = 3, label = labels, xlabel=L"$t$",
    ylabel=L"$MSE_{95}$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p4 = StatsPlots.plot(times[101:end], misef[100:end, :], lw = 3, label = labels, xlabel=L"$t$",
    ylabel=L"MISE", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p5 = StatsPlots.plot(times[2:end], E, lw = 3, label = labels, xlabel=L"$t$",
    ylabel=L"$E(\rho_t)$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p6 = StatsPlots.plot();
plot(p2, p3, p4, p5, layout = (2, 2))

savefig(p2, "initial_distribution_variance.pdf")
savefig(p3, "initial_distribution_mse.pdf")
savefig(p4, "initial_distribution_mise.pdf")
savefig(p5, "initial_distribution_E.pdf")
