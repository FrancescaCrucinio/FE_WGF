push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
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
using JLD;
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

# entropy function
function remove_non_finite(x)
       return isfinite(x) ? x : zero(x)
end
# Shepp Logan phantom
phantom = readdlm("PET/phantom.txt", ',', Float64);
phantom = reverse(phantom, dims=1);
pixels = size(phantom);
# entropy
phantom_ent = -mean(remove_non_finite.(phantom .* log.(phantom)));
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

# dt and number of iterations
dt = 1e-03;
Niter = 1000;
# samples from h(y)
M = 500;
# number of particles
Nparticles = 500;
# regularisation parameter
alpha = 0.001;
# variance of normal describing alignment
sigma = 0.02;
# sample from μ
muSample = histogram2D_sampler(sinogram, phi_angle, xi, 10^6);

# grid
Xbins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
Ybins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
gridX = repeat(Xbins, inner=[pixels[2], 1]);
gridY = repeat(Ybins, outer=[pixels[1] 1]);
KDEeval = [gridX gridY];
R"""
    data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = c($phantom));
    p <- ggplot(data, aes(x, y)) +
        geom_raster(aes(fill = z), interpolate=TRUE) +
        theme_void() +
        theme(legend.position = "none", aspect.ratio=1) +
        scale_fill_viridis(discrete=FALSE, option="magma")
    # ggsave("phantom.eps", p)
"""

# WGF
x, y = wgf_pet_tamed(Nparticles, dt, Niter, alpha, muSample, M, phi_angle, xi, sigma, 0.5);

# function computing KDE
function psi(t)
    RKDE = rks.kde(x = [t[1:Nparticles] t[(Nparticles+1):(2Nparticles)]], var"eval.points" = KDEeval);
    return abs.(rcopy(RKDE[3]));
end
# function computing entropy
function psi_ent(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
end

# KDE
KDEyWGF = mapslices(psi, [x y], dims = 2);
# entropy
ent = mapslices(psi_ent, KDEyWGF, dims = 2);
# last time step
KDEyWGFfinal = KDEyWGF[9150, :];
plot(1:Niter, ent)
hline!([phantom_ent])

# plot
R"""
    # solution
    data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = $KDEyWGFfinal);
    p <- ggplot(data, aes(x, y)) +
        geom_raster(aes(fill = z), interpolate=TRUE) +
        theme_void() +
        theme(legend.position = "none", aspect.ratio=1) +
        scale_fill_viridis(discrete=FALSE, option="magma")
    # ggsave("pet.eps", p)
"""
# ise
petWGF = reshape(KDEyWGFfinal, (pixels[1], pixels[2]));
var(petWGF .- phantom)


save("pet18Oct2020.jld", "alpha", alpha, "dt", dt, "Nparticles", Nparticles,
    "Niter", Niter, "KDEyWGF", KDEyWGF);

Nparticles = load("pet18Oct2020.jld", "Nparticles");
KDEyWGF = load("pet18Oct2020.jld", "KDEyWGF");
