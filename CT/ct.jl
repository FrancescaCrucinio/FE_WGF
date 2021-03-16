push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
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
using Interpolations;
using Images;
using Distances;
using KernelDensity;
# R
using RCall;
R"""
library(ggplot2)
library(scales)
library(viridis)
"""
# custom packages
using wgf_prior;
using samplers;

# set seed
Random.seed!(1234);

# data image
sinogram = readdlm("CT/sinogramCT.txt", ',', Float64);
pixels = size(sinogram, 2);
# sinogram = reverse(sinogram, dims=1);
# number of angles
nphi = size(sinogram, 2);
# angles
phi_angle = range(0, stop = 2*pi, length = nphi);
# number of offsets
offsets = floor(size(sinogram, 1)/2);
xi = range(-offsets, stop = offsets, length = size(sinogram, 1));
xi = xi/maximum(xi);

# grid
X1bins = range(-1 + 1/pixels[1], stop = 1 - 1/pixels[1], length = pixels[1]);
X2bins = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
gridX1 = repeat(X1bins, inner=[pixels[2], 1]);
gridX2 = repeat(X2bins, outer=[pixels[1] 1]);
KDEeval = [gridX1 gridX2];

######## rewrite the function below -- we have the empirical measure not the sample!!!


# functional approximation
function psi(piSample)
    loglik = zeros(1, size(muSample, 1));
    for i=1:size(muSample, 1)
        loglik[i] = mean(pdf.(Normal.(0, sigma), piSample[:, 1] * cos(muSample[i, 1]) .+
            piSample[:, 2] * sin(muSample[i, 1]) .- muSample[i, 2]));
    end
    loglik = -log.(loglik);
    kl = mean(loglik);
    prior = pdf.(MvNormal(m0, sigma0), piSample);
    piKDE = kde((piSample[:, 1],  piSample[:, 2]));
    pihat = pdf(piKDE, piSample[:, 1], piSample[:, 2]);
    pihat = abs.(pihat[:]);
    kl_prior = mean(log.(pihat./prior));
    return kl+alpha*kl_prior;
end

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-2;
# number of iterations
Niter = 100;
# regularisation parameter
alpha = 0.0017;
# prior mean
m0 = [0; 0];
sigma0 = [0.1; 0.1];
# initial distribution
x0 = sigma0[1]*randn(2, Nparticles);
# variance of normal describing alignment
sigma = 0.02;

# WGF
tWGF = @elapsed begin
x1, x2 = wgf_ct_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, M, sinogram, phi_angle, xi, sigma);
end
