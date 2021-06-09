# push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using JLD;
# custom packages
using diagnostics;
using wgfserver;

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
Niter = 100;
# samples from h(y)
M = 500;
# values at which evaluate h(y)
refY = range(0, stop = 1, length = 1000);
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 500;
# regularisation parameters
lambda = range(0.001, stop = 0.1, length = 100);
# number of repetitions
Nrep = 1000;

# diagnostics
diagnosticsWGF = zeros(length(lambda), 7);
Threads.@threads for i=1:length(lambda)
    # mise, mean and variance
    drepWGF = zeros(Nrep, 7);
    @simd for j=1:Nrep
        # initial distribution
        x0 = rand(1)*ones(1, Nparticles);
        # run WGF
        x, _ = wgf_AT(Nparticles, dt, Niter, lambda[i], x0, M);
        KDEyWGF = KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=bwnormal(x[end,:]));
        m, v, mse95, mise, kl, ent = diagnosticsALL(f, h, g, KDEx, KDEyWGF, refY);
        drepWGF[j, :] = [m, v, mse95 , mise, kl - lambda[i]*ent, ent, kl];
        println("$i, $j")
    end
    diagnosticsWGF[i, :] = mean(drepWGF,dims = 1);
end


save("parameters500zoom.jld", "lambda", lambda, "diagnosticsWGF", diagnosticsWGF,
   "Nparticles", Nparticles, "Niter", Niter)
