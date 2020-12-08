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

# set seed
Random.seed!(1234);

# data image
sinogram = readdlm("PET/sinogram.txt", ',', Float64);
# number of angles
nphi = size(sinogram, 2);
# angles
phi_angle = range(0, stop = 2*pi, length = nphi);
# number of offsets
offsets = floor(size(sinogram, 1)/2);
xi = range(-offsets, stop = offsets, length = size(sinogram, 1));

# sample from μ
muSample = histogram2D_sampler(sinogram, phi_angle, xi, 10^6);
# dt and number of iterations
dt = 1e-02;
Niter = 200;
# samples from h(y)
M = 5000;
# number of particles
Nparticles = 5000;
# regularisation parameter
alpha = range(0.001, stop = 1, length = 10);
# variance of normal describing alignment
sigma = 0.02;

# function computing KDE
# grid
Xbins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
Ybins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
gridX = repeat(Xbins, inner=[pixels[2], 1]);
gridY = repeat(Ybins, outer=[pixels[1] 1]);
KDEeval = [gridX gridY];
function phi(t)
    RKDE = rks.kde(x = [t[1:Nparticles] t[(Nparticles+1):(2Nparticles)]], var"eval.points" = KDEeval);
    return abs.(rcopy(RKDE[3]));
end
a = 1;
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueH = @rget sinogram;
    refY1 = @rget phi_angle;
    refY2 = @rget xi;
    # approximated value
    delta1 = refY1[2] - refY1[1];
    delta2 = refY2[2] - refY2[1];
    hatH = zeros(length(refY2), length(refY1));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY2)
        for j=1:length(refY1)
            hatH[i, j] = delta1*delta2*sum(pdf.(Normal.(0, sigma), KDEeval[:, 1] * cos(refY1[j]) .+
                KDEeval[:, 2] * sin(refY1[j]) .- refY2[i]).*t);
        end
    end
    kl = kl_divergence(trueH[:], hatH[:]);
    return kl-a*ent;
end

# divide muSample into groups
L = 10;
muSample = reshape(muSample, (L, Int(length(muSample)/L)));
E = zeros(length(alpha), L);
Threads.@threads for i=1:length(alpha)
    @simd for l=1:L
        # get reduced sample
        muSampleL = muSample[1:end .!= l, :];
        # WGF
        x, y = wgf_pet_tamed(Nparticles, dt, Niter, alpha[i], muSampleL, M, phi_angle, xi, sigma, 0.5);
        # KL
        a = alpha[i];
        KDE = phi([x[Niter, :], y[Niter, :]]);
        E[i, l] = psi(KDE);
        println("$i, $l")
    end
end
plot(alpha,  mean(E, dims = 2))
