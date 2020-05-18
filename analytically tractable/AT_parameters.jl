push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
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
using wgf;

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
M = 1000;
# values at which evaluate h(y)
refY = range(0, stop = 1, length = 1000);
# values at which evaluate KDE
KDEx = range(0, stop = 1, length = 1000);
# number of particles
Nparticles = 1000;
# regularisation parameters
lambda = range(0.001, stop = 1, length = 100);
# number of repetitions
Nrep = 1000;

# diagnostics
diagnosticsWGF = zeros(length(lambda), 7);
Threads.@threads for i=1:length(lambda)
    # mise, mean and variance
    drepWGF = zeros(Nrep, 7);
    @simd for j=1:Nrep
        println("$i, $j")
        # initial distribution
        x0 = rand(1)*ones(1, Nparticles);
        # run WGF
        x, _ = wgf_AT(Nparticles, dt, Niter, lambda[i], x0, M);
        KDEyWGF = KernelEstimator.kerneldensity(x[end,:], xeval=KDEx, h=bwnormal(x[end,:]));
        m, v, mse95, mise, kl, ent = diagnosticsALL(f, h, g, KDEx, KDEyWGF, refY);
        drepWGF[j, :] = [m, v, mse95 , mise, kl - lambda[i]*ent, ent, kl];
    end
    diagnosticsWGF[i, :] = mean(drepWGF,dims = 1);
end

# lambda = load("parametersN1000resampling.jld", "lambda");
# diagnosticsWGF = load("parametersN1000resampling.jld", "diagnosticsWGF");

# pyplot()
# p1 = plot(lambda, diagnosticsWGF[:, 1], lw = 3, legend = false,
#         xlabel="lambda", ylabel="mean");
# hline!([0.5]);
# p2 = plot(lambda, diagnosticsWGF[:, 2], lw = 3, legend = false,
#         xlabel="lambda", ylabel="variance");
# hline!([0.043^2]);
# p3 = plot(lambda, diagnosticsWGF[:, 3], lw = 3, legend = false,
#         xlabel="lambda", ylabel="95th MSE");
# p4 = plot(lambda, diagnosticsWGF[:, 4], lw = 3, legend = false,
#         xlabel="lambda", ylabel="MISE");
# p5 = plot(lambda, diagnosticsWGF[:, 5], lw = 3, legend = false,
#         xlabel="lambda", ylabel="E(rho)");
# p6 = plot(lambda, diagnosticsWGF[:, 6], lw = 3, legend = false,
#         xlabel="lambda", ylabel="entropy");
# plot(p1, p2, p3, p4, p5, p6, layout = (2, 3))

# savefig(p1, "mean1000.pdf")
# savefig(p2, "var1000.pdf")
# savefig(p3, "mse1000.pdf")
# savefig(p4, "mise1000.pdf")
# savefig(p5, "e1000.pdf")
# savefig(p6, "entropy1000.pdf")
#save("C:/Users/francesca/Dropbox/parameters.jld", "lambda", lambda, "diagnosticsWGF", diagnosticsWGF,
    # "Nparticles", Nparticles, "Niter", Niter)
