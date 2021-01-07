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
using Distances;
using Images;
# R
using RCall;
@rimport ks as rks;
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

# Shepp Logan phantom
phantom = readdlm("PET/phantom.txt", ',', Float64);
phantom = reverse(phantom, dims=1);
pixels = size(phantom);
# data image
sinogram = readdlm("PET/sinogram.txt", ',', Float64);
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
X1bins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
X2bins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
gridX1 = repeat(X1bins, inner=[pixels[2], 1]);
gridX2 = repeat(X2bins, outer=[pixels[1] 1]);
KDEeval = [gridX1 gridX2];

# function computing KDE
function phi(t)
    RKDE = rks.kde(x = [t[1:Nparticles] t[(Nparticles+1):(2Nparticles)]], var"eval.points" = KDEeval);
    return abs.(rcopy(RKDE[3]));
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
# dt and number of iterations
dt = 1e-02;
Niter = 50;
# samples from h(y)
M = 5000;
# number of particles
Nparticles = 5000;
# regularisation parameter
alpha = 0.0001;
# variance of normal describing alignment
sigma = 0.02;
# sample from μ
muSample = histogram2D_sampler(sinogram, phi_angle, xi, 10^6);

x1, x2 = wgf_pet_tamed(Nparticles, dt, Niter, alpha, muSample, M, sigma, 0.5);

# KDE
KDEyWGF = mapslices(phi, [x1 x2], dims = 2);
# entropy
ent = mapslices(psi_ent, KDEyWGF, dims = 2);
# KL
KLWGF = mapslices(psi_kl, KDEyWGF, dims = 2);

# plot
plot(KLWGF)
phantom_ent = psi_ent(phantom);
plot(ent)
hline!([phantom_ent])
plot(KLWGF .- alpha * ent)

KDEyWGFfinal = rks.kde(x = [x1[end, :] x2[end, :]], var"eval.points" = KDEeval);
KDEyWGFfinal = abs.(rcopy(KDEyWGFfinal[3]));
# plot
R"""
    # phantom
    data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = c($phantom));
    p1 <- ggplot(data, aes(x, y)) +
        geom_raster(aes(fill = z), interpolate=TRUE) +
        theme_void() +
        theme(legend.position = "none", aspect.ratio=1) +
        scale_fill_viridis(discrete=FALSE, option="magma")
    # ggsave("phantom.eps", p1)
    # solution
    data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = $KDEyWGFfinal);
    p2 <- ggplot(data, aes(x, y)) +
        geom_raster(aes(fill = z), interpolate=TRUE) +
        theme_void() +
        theme(legend.position = "none", aspect.ratio=1) +
        scale_fill_viridis(discrete=FALSE, option="magma")
    # ggsave("pet.eps", p2)
"""
# ise
petWGF = reshape(KDEyWGFfinal, (pixels[1], pixels[2]));
petWGF = reverse(petWGF, dims=1);
petWGF = petWGF/maximum(petWGF);
var(petWGF .- phantom)
