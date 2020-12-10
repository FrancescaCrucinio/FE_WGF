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
Xbins = range(-0.75+ 1/pixels, stop = 0.75 - 1/pixels, length = pixels);
Ybins = range(-0.75 + 1/pixels, stop = 0.75 - 1/pixels, length = pixels);
gridX = repeat(Xbins, inner=[pixels, 1]);
gridY = repeat(Ybins, outer=[pixels 1]);
KDEeval = [gridX gridY];

# WGF
x, y = wgf_pet_tamed(Nparticles, dt, Niter, alpha, muSample, M, sigma, 0.5);

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
KDEyWGFfinal = psi([x[end, :] y[end, :]]);
plot(ent)

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
