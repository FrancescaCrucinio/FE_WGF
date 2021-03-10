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
using wgf;
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

# dt and number of iterations
dt = 1e-02;
Niter = 100;
# samples from h(y)
M = 10000;
# number of particles
Nparticles = 10000;
# regularisation parameter
alpha = 0.001;
# variance of normal describing alignment
sigma = 0.01;
# sample from μ
muSample = histogram2D_sampler(sinogram, phi_angle, xi, 10^6);

# grid
dx = 100;
X1bins = range(-0.75+ 1/dx, stop = 0.75 - 1/dx, length = dx);
X2bins = range(-0.75 + 1/dx, stop = 0.75 - 1/dx, length = dx);
gridX1 = repeat(X1bins, inner=[dx, 1]);
gridX2 = repeat(X2bins, outer=[dx 1]);
KDEeval = [gridX1 gridX2];
# function computing KDE
function phi(t)
    B = kde((t[1:Nparticles], t[(Nparticles+1):(2Nparticles)]));
    Bpdf = pdf(B, X1bins, X2bins);
    return abs.(Bpdf[:]);
end
# function computing entropy
function psi_ent(t)
    t = t./maximum(t);
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
end
# function computing KL
function psi_kl(t)
    # kl
    trueMu = sinogram;
    refY1 = phi_angle;
    refY2 = xi;
    # approximated value
    delta1 = refY1[2] - refY1[1];
    delta2 = refY2[2] - refY2[1];
    hatMu = zeros(length(refY2), length(refY1));
    # convolution with approximated ρ
    # this gives the approximated value
    for i=1:length(refY2)
        for j=1:length(refY1)
            hatMu[i, j] = sum(pdf.(Normal.(0, sigma), KDEeval[:, 1] * cos(refY1[j]) .+
                KDEeval[:, 2] * sin(refY1[j]) .- refY2[i]).*t);
        end
    end
    hatMu = hatMu/maximum(hatMu);
    kl = kl_divergence(trueMu[:], hatMu[:]);
    return kl;
end

# WGF
x, y = wgf_pet_tamed(Nparticles, dt, Niter, alpha, muSample, M, sigma, 0.5);

# KDE
KDEyWGF = mapslices(phi, [x y], dims = 2);
# entropy
ent = mapslices(psi_ent, KDEyWGF, dims = 2);
# KL
KLWGF = mapslices(psi_kl, KDEyWGF, dims = 2);

# plot
plot(KLWGF)
plot(ent)
plot(KLWGF .- alpha * ent)

# last time step
KDEyWGFfinal = psi([x[end, :] y[end, :]]);

# plot
R"""
    # solution
    data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = $KDEyWGFfinal);
    p <- ggplot(data, aes(x, y)) +
        geom_raster(aes(fill = z), interpolate=TRUE) +
        theme_void() +
        theme(legend.position = "none", aspect.ratio=1) +
        scale_fill_gradient(low = "black", high = "white")
    # ggsave("ct.eps", p)
"""
# ctWGF = reshape(KDEyWGFfinal, (pixels, pixels));
# res = map(clamp01nan, Gray.(ctWGF))
# save("ct.png", Gray.(ctWGF))
