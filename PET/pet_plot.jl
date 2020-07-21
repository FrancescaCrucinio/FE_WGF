push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using DelimitedFiles;
using JLD;
using RCall;
@rimport ks as rks
# custom modules
using diagnostics;

# entropy function
function remove_non_finite(x)
       return isfinite(x) ? x : zero(x)
end
# Shepp Logan phantom
phantom = readdlm("PET/phantom.txt", ',', Float64);
phantom = reverse(phantom, dims=1);
# entropy
phantom_ent = -mean(remove_non_finite.(phantom .* log.(phantom)));
pixels = size(phantom);
# SMC-EMS reconstruction
petSMCEMS = readdlm("PET/pet_smcems.txt", ',', Float64);
# entropy
petSMCEMS_ent = -mean(remove_non_finite.(petSMCEMS .* log.(petSMCEMS)));
# data image
sinogram = readdlm("PET/sinogram.txt", ',', Float64);
sinogram = reverse(sinogram, dims=1);
# number of angles
nphi = size(sinogram, 2);
# angles
phi = range(0, stop = 2*pi, length = nphi);
# number of offsets
offsets = floor(size(sinogram, 1)/2);
xi = range(-offsets, stop = offsets, length = size(sinogram, 1));

x = load("PET/pet15062020.jld", "x");
y = load("PET/pet15062020.jld", "y");
Niter = load("PET/pet15062020.jld", "Niter");
alpha = load("PET/pet15062020.jld", "lambda");

# grid
Xbins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
Ybins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
gridX = repeat(Xbins, inner=[pixels[2], 1]);
gridY = repeat(Ybins, outer=[pixels[1] 1]);
KDEeval = [gridX gridY];


#
# # select which steps to show
# showIter = [1, 10, 50, 100, 300, 500];
# Npic = 6;
# # mise
# miseWGF = zeros(1, Npic);
# petWGF_ent = zeros(1, Npic);
#
# # grid
# Xbins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
# Ybins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
# gridX = repeat(Xbins, inner=[pixels[2], 1]);
# gridY = repeat(Ybins, outer=[pixels[1] 1]);
# KDEeval = [gridX gridY];
# # non-zero entries of phantom
# phantom_pos = (phantom.>0);
# for n=1:Npic
#     # KDE
#     KDEdata = [x[showIter[n], :] y[showIter[n], :]];
#     KDEyWGF = rks.kde(x = KDEdata, var"eval.points" = KDEeval);
#     petWGF = reshape(rcopy(KDEyWGF[3]), (pixels[1], pixels[2]));
#     # plot
#     R"""
#     library(ggplot2)
#     library(scales)
#     library(viridis)
#         data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = $KDEyWGF[3]);
#         p <- ggplot(data, aes(x, y)) +
#             geom_raster(aes(fill = estimate), interpolate=TRUE) +
#             theme_void() +
#             theme(legend.position = "none", aspect.ratio=1) +
#             scale_fill_viridis(discrete=FALSE, option="magma")
#         # ggsave(paste("pet", $n, ".eps", sep=""), p)
#     """
#     # mise
#     miseWGF[n] = (norm(petWGF - reverse(phantom, dims=1)).^2)/length(petWGF);
#     # entropy
#     petWGF_ent[n] = -mean(remove_non_finite.(petWGF .* log.(petWGF)));
#     # relative error
#     rel_error = relative_error(petWGF, reverse(phantom, dims = 1));
#     R"""
#         data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = c($rel_error));
#         p <- ggplot(data, aes(x, y)) +
#             geom_raster(aes(fill = z), interpolate=TRUE) +
#             theme_void() +
#             theme(legend.position = "none", aspect.ratio=1) +
#             scale_fill_viridis(discrete=FALSE, option="magma")
#         ggsave(paste("pet_relerror", $n, ".eps", sep=""), p)
#     """
# end
# R"""
#     data <- data.frame(x = $KDEeval[, 1], y = $KDEeval[, 2], z = c($phantom));
#     p <- ggplot(data, aes(x, y)) +
#         geom_raster(aes(fill = z), interpolate=TRUE) +
#         theme_void() +
#         theme(legend.position = "none", aspect.ratio=1) +
#         scale_fill_viridis(discrete=FALSE, option="magma")
#     ggsave("phantom.eps", p)
#     data <- data.frame(x = $xi, y = $phi, z = c($sinogram));
#     p <- ggplot(data, aes(x, y)) +
#         geom_raster(aes(fill = z), interpolate=TRUE) +
#         theme_void() +
#         theme(legend.position = "none", aspect.ratio=1) +
#         scale_fill_viridis(discrete=FALSE, option="magma")
#     ggsave("sinogram.eps", p)
# """
