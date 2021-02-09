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
using KernelDensity;
# R
using RCall;
@rimport ks as rks;
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

# sample from μ
muSample = histogram2D_sampler(sinogram, phi_angle, xi, 10^6);

# grid
X1bins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
X2bins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
gridX1 = repeat(X1bins, inner=[pixels[2], 1]);
gridX2 = repeat(X2bins, outer=[pixels[1] 1]);
KDEeval = [gridX1 gridX2];

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
Nparticles = 1000;
# number of samples from μ to draw at each iteration
M = 1000;
# time discretisation
dt = 1e-2;
# number of iterations
Niter = 100;
# regularisation parameter
alpha = 0.0017;
# initial distribution
x0 =
# prior mean
m0 =
sigma0 =
# variance of normal describing alignment
sigma = 0.01;

# WGF
tWGF = @elapsed begin
x1, x2 = wgf_ct_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M, sigma, 0.5);
end

# check convergence
EWGF = mapslices(psi, [x1 x2], dims = 2);
plot(EWGF)

# KDE
piKDE = kde((x1[Niter, :],  x2[Niter, :]));
KDEyWGF = pdf(piKDE, X1bins, X2bins);
KDEyWGF = abs.(KDEyWGF[:]);

# plot
R"""
    # solution
    data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = $KDEyWGF);
    p2 <- ggplot(data, aes(x, y)) +
        geom_raster(aes(fill = z), interpolate=TRUE) +
        theme_void() +
        theme(legend.position = "none", aspect.ratio=1) +
        scale_fill_viridis(discrete=FALSE, option="magma")
    # ggsave("ct.eps", p2)
"""
