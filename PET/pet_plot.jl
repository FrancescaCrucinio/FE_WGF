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
using KernelDensity;
using Interpolations;
using JLD;
using Plots;
# custom modules
using diagnostics;

pyplot()
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

x = load("PET/pet15062020.jld", "x");
y = load("PET/pet15062020.jld", "y");
Niter = load("PET/pet15062020.jld", "Niter");


# select which steps to show
showIter = [1, 10, 50, 100, 300, 500];
Npic = 6;
# mise
miseWGF = zeros(1, Npic);
petWGF_ent = zeros(1, Npic);
# plots
p = repeat([plot(1)], Npic);
p_relative_error = repeat([plot(1)], Npic);
# grid
Xbins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
Ybins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
# non-zero entries of phantom
phantom_pos = (phantom.>0);
for n=1:Npic
    # KDE
    # swap x and y for KDE function (scatter plot shows that x, y are correct)
    KDEyWGF =  KernelDensity.kde((y[showIter[n], :], x[showIter[n], :]));
    petWGF = pdf(KDEyWGF, Ybins, Xbins);
    p[n] = heatmap(Xbins, Ybins, petWGF, legend = :none,
        aspect_ratio=1, showaxis=false, grid=false, size = (128, 128));
    # mise
    miseWGF[n] = (norm(petWGF - reverse(phantom, dims=1)).^2)/length(petWGF);
    # entropy
    petWGF_ent[n] = -mean(remove_non_finite.(petWGF .* log.(petWGF)));
    # relative error
    rel_error = relative_error(petWGF, reverse(phantom, dims = 1));
    p_relative_error[n] = heatmap(Xbins, Ybins, rel_error, legend = :none,
        aspect_ratio=1, showaxis=false, grid=false, size = (128, 128));
    savefig(p[n], "pet$n.pdf");
    savefig(p_relative_error[n], "pet_relerror$n.pdf");
end
p_phantom = heatmap(reverse(phantom, dims=1), legend = :none,
    aspect_ratio=1, showaxis=false, grid=false, size = (128, 128));
p_sinogram = heatmap(reverse(sinogram, dims=1), legend = :none,
    showaxis=false, grid=false, size = (128, 185));
plot(p..., layout=(2, 3), showaxis=false)
plot(p_relative_error..., layout=(2, 3), showaxis=false)

savefig(p_phantom, "phantom.pdf")
savefig(p_sinogram, "sinogram.pdf")
