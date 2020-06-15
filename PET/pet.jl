# push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using DelimitedFiles;
using KernelDensity;
using Interpolations;
using JLD;
# custom packages
using diagnostics;
using wgf;

# entropy function
function remove_non_finite(x)
       return isfinite(x) ? x : zero(x)
end
# Shepp Logan phantom
phantom = readdlm("PET/phantom.txt", ',', Float64);
# entropy
phantom_ent = -mean(remove_non_finite.(phantom .* log.(phantom)));
pixels = size(phantom);
# SMC-EMS reconstruction
petSMCEMS = readdlm("PET/pet_smcems.txt", ',', Float64);
# entropy
petSMCEMS_ent = -mean(remove_non_finite.(petSMCEMS .* log.(petSMCEMS)));
# data image
sinogram = readdlm("PET/sinogram.txt", ',', Float64)
# number of angles
nphi = size(sinogram, 2);
# angles
phi = range(0, stop = 2*pi, length = nphi);
# number of offsets
offsets = floor(size(sinogram, 1)/2);
xi = range(-offsets, stop = offsets, length = size(sinogram, 1));

# dt and number of iterations
dt = 1e-03;
Niter = 500;
# samples from h(y)
M = 20000;
# number of particles
Nparticles = 20000;
# regularisation parameter
lambda = 0.01;
# variance of normal describing alignment
sigma = 0.02;

# WGF
x, y = wgf_pet(Nparticles, dt, Niter, lambda, sinogram, M, phi, xi, sigma);

# KDE
# swap x and y for KDE function (scatter plot shows that x, y are correct)
KDEyWGF =  KernelDensity.kde((y[end, :], x[end, :]));
Xbins = range(-1 + 1/pixels[1], stop = 1 - 1/pixels[1], length = pixels[1]);
Ybins = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
petWGF = pdf(KDEyWGF, Ybins, Xbins);
# entropy
petWGF_ent = -mean(remove_non_finite.(petWGF .* log.(petWGF)));

p = heatmap(Xbins, Ybins, petWGF)


# savefig(p, "pet.pdf")

miseWGF = (norm(petWGF - phantom).^2)/length(petWGF);
miseSMCEMS = (norm(petSMCEMS - phantom).^2)/length(petSMCEMS);

x = load("PET/pet15062020.jld", "x");
y = load("PET/pet15062020.jld", "y");
