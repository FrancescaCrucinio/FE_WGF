push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
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

# plot
p1 = StatsPlots.plot(1:Niter-1, E, lw = 3)
p2 = StatsPlots.plot(1:Niter-1, m, lw = 3)
p3 = StatsPlots.plot(1:Niter-1, v, lw = 3)
p4 = StatsPlots.plot(1:Niter-1, q, lw = 3)
p5 = StatsPlots.plot(1:Niter-1, misef, lw = 3)
p6 = StatsPlots.plot()
plot(p1, p2, p3, p4, p5, p6, layout = (3, 2))
