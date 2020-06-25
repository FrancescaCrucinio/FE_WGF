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
using wgf;
using samplers;

Imageh = convert(Array{Float64}, Imageh);

# number of iterations
Niter = 100;
# samples from h(y)
M = 5000;
# number of particles
Nparticles = 5000;
# regularisation parameter
lambda = 0.01;
dt = 10^-3;

sigma = 0.02;
a = -100;
x, y = wgf_deblurring(Nparticles, Niter, dt, lambda, Imageh, M, sigma, a);

scatter(x[end, :], y[end, :])
