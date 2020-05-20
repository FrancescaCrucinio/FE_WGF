push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelDensity;
using Random;
using JLD;
using LaTeXStrings;
using QuadGK;
# custom packages
using diagnostics;
using wgf;

pyplot()
# set seed
Random.seed!(1234);

# data for 2D analytic example
# integral sine function
Si(x) = quadgk(t -> sin(t)/t, 0, x)[1];
f(x) = (cos.(x[1].*x[2]) .* (x[1].^2 .+ x[2].^2 .- x[1] .- x[2] .+ 2/3))./
    (Si(1)*2/3-2+2*sin(1));
h(y) = ((y[1].^2 .+ y[2].^2)*Si(1) .- 2*(1-cos(1))*(y[1] .+ y[2]) + 2*sin(1) - 2*cos(1))/
        (Si(1)*2/3-2+2*sin(1));
g(x, y) = ((y[1] .- x[1]).^2 .+ (y[2] .- x[2]).^2)./
    (x[1].^2 .+ x[2].^2 .- x[1] .- x[2] .+ 2/3);

# dt and number of iterations
dt = 1e-03;
Niter = 2000;

# samples from h(y)
M = 1000;
# number of particles
Nparticles = 1000;
# regularisation parameter
lambda = 0.05;

# initial distribution
x0 = rand(2, Nparticles);
# run WGF
x, y = wgf_2D_analytic(Nparticles, dt, Niter, lambda, x0, M);
KDEyWGF =  KernelDensity.kde((x[end, :], y[end, :]));
Xbins = range(0, stop = 1, length = 1000);
Ybins = range(0, stop = 1, length = 1000);
res = pdf(KDEyWGF, Ybins, Xbins);
p1 = heatmap(Xbins, Ybins, res);

fplot = zeros(1000, 1000);
for i=1:1000
    for j=1:1000
        fplot[j, i] = f([Xbins[i]; Ybins[j]]);
    end
end
p2 = heatmap(Xbins, Ybins, fplot);
