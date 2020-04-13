push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
# using ImageMagick;
# using TestImages, Colors;
# using Images;
using LinearAlgebra;
using DelimitedFiles;
using KernelDensity;
using Interpolations;
# custom packages
using diagnostics;
using smcems;
using wgf;
using samplers;

# Shepp Logan phantom
phantom = readdlm("phantom.txt", ',', Float64);
pixels = size(phantom);
# data image
sinogram = readdlm("sinogram.txt", ',', Float64)
# number of angles
nphi = size(sinogram, 2);
# angles
phi = range(0, stop = 2*pi, length = nphi);
# number of offsets
offsets = floor(size(I, 1)/2);
xi = range(-offsets, stop = offsets, length = size(sinogram, 1));

# number of iterations
Niter = trunc(Int, 1e03);
# samples from h(y)
M = 10000;
# number of particles
Nparticles = 10000;
# regularisation parameter
lambda = 25;
# variance of normal describing alignment
sigma = 0.02;

# WGF
x, y = wgf_pet(Nparticles, Niter, lambda, sinogram, M, phi, xi, sigma);

# KDE
# swap x and y for KDE function (scatter plot shows that x, y are correct)
KDEyWGF =  KernelDensity.kde((y[end, :], x[end, :]));
Xbins = range(-1 + 1/pixels[1], stop = 1 - 1/pixels[1], length = pixels[1]);
Ybins = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
res = pdf(KDEyWGF, Ybins, Xbins);
p = heatmap(Xbins, Ybins, res)

# savefig(p, "pet.pdf")

mise = (norm(res - phantom).^2)/length(res)
