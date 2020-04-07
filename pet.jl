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
using DelimitedFiles;
using KernelDensity;
using Interpolations;
# custom packages
using diagnostics;
using smcems;
using wgf;
using samplers;

I = readdlm("sinogram.txt", ',', Float64)
pixels = size(I);

phi = range(0, stop = 2*pi, length = pixels[2]);
offsets = floor(pixels[1]/2);
xi = range(-offsets, stop = offsets, length = pixels[1]);
# number of iterations
Niter = trunc(Int, 1e03);
# samples from h(y)
M = 1000;
# number of particles
Nparticles = 5000;
# regularisation parameter
lambda = 50;

sigma = 0.02;

x, y = wgf_pet(Nparticles, Niter, lambda, I, M, phi, xi, sigma);

KDEyWGF =  KernelDensity.kde((x[end, :], y[end, :]));
Xbins = range(-1, stop = 2, length = 1000);
Ybins = range(-1, stop = 2, length = 1000);
res = pdf(KDEyWGF, Ybins, Xbins);
heatmap(Xbins, Ybins, res)
