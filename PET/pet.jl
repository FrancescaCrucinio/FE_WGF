# push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/myModules")
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
using KernelDensity;
using Interpolations;
using JLD;
# custom packages
using diagnostics;
using wgf;

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

# dt and number of iterations
dt = 1e-03;
Niter = 500;
# samples from h(y)
M = 20000;
# number of particles
Nparticles = 20000;
# regularisation parameter
lambda = 0.0005;
# variance of normal describing alignment
sigma = 0.02;

# WGF
x, y = wgf_pet(Nparticles, dt, Niter, lambda, sinogram, M, phi, xi, sigma);


# grid
Xbins = range(-0.75+ 1/pixels[1], stop = 0.75 - 1/pixels[1], length = pixels[1]);
Ybins = range(-0.75 + 1/pixels[2], stop = 0.75 - 1/pixels[2], length = pixels[2]);
gridX = repeat(Xbins, inner=[pixels[2], 1]);
gridY = repeat(Ybins, outer=[pixels[1] 1]);
KDEeval = [gridX gridY];

# function computing KDE
function psi(t)
    RKDE = rks.kde(x = [t[1:N]; t[(N+1):(2N)]], var"eval.points" = KDEeval);
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


KDEyWGF = mapslices(psi, [x y], dims = 2);
ent = mapslices(psi_ent, KDEyWGF, dims = 2);
plot(1:Niter, ent)
#

save("pet20062020.jld", "lambda", lambda, "x", x,
   "y", y, "Niter", Niter, "Nparticles", Nparticles, "M", M, "dt", dt);
