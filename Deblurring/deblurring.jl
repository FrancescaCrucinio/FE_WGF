push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using ImageMagick;
using TestImages, Colors;
using Images;
# custom packages
using diagnostics;
using smcems;
using wgf;
using samplers;

I = load("BCblurred.png");
I = Gray.(I);
I = convert(Array{Float64}, I);
pixels = size(I);

# number of iterations
Niter = trunc(Int, 1e04);
# samples from h(y)
M = 1000;
# number of particles
Nparticles = 5000;
# regularisation parameter
lambda = 50;

sigma = 0.02;
a = 1.01;
b = 1.01;
v = 128;
x, y = wgf_deblurring(Nparticles, Niter, lambda, I, M, sigma, v, a, b);

scatter(x[Niter, :], y[Niter, :])
