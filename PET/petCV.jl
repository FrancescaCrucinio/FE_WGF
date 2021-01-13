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

# data image
sinogram = readdlm("PET/sinogram.txt", ',', Float64);
# number of angles
pixels = size(sinogram, 2);
# angles
phi_angle = range(0, stop = 2*pi, length = pixels);
# number of offsets
offsets = floor(size(sinogram, 1)/2);
xi = range(-offsets, stop = offsets, length = size(sinogram, 1));
xi = xi/maximum(xi);

# grid
X1bins = range(-0.75+ 1/pixels, stop = 0.75 - 1/pixels, length = pixels);
X2bins = range(-0.75 + 1/pixels, stop = 0.75 - 1/pixels, length = pixels);
gridX1 = repeat(X1bins, inner=[pixels, 1]);
gridX2 = repeat(X2bins, outer=[pixels 1]);
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
Niter = 30;
# samples from h(y)
M = 5000;
# number of particles
Nparticles = 5000;
# regularisation parameter
alpha = range(0.0002, stop = 0.0009, length = 10);
# variance of normal describing alignment
sigma = 0.02;
# sample from μ
muSample = histogram2D_sampler(sinogram, phi_angle, xi, 10^6);

L = 10;

ent = zeros(length(alpha), L);
kl = zeros(length(alpha), L);
for i=1:length(alpha)
    for l=1:L
        # get reduced sample
        muSampleL = muSample[setdiff(1:10^6, Tuple(((1:10^5) .+ (l-1)*10^5))), :];
        # WGF
        x1, x2 = wgf_pet_tamed(Nparticles, dt, Niter, alpha[i], muSampleL, M, sigma, 0.5);
        # KL
        KDE = phi([x1[Niter, :]; x2[Niter, :]]);
        ent[i, l] = psi_ent(KDE);
        kl[i, l] = psi_kl(KDE);
        println("$i, $l")
    end
end

E = zeros(length(alpha), L);
for i=1:length(alpha)
    for j=1:L
        E[i, j] = kl[i, j] - alpha[i]*ent[i, j];
    end
end

plot(alpha,  mean(E, dims = 2))
