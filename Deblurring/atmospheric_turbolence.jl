# push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/Github/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using KernelDensity;
using Interpolations;
using FileIO;
using ImageIO;
using JLD;
using Images;
using RCall;
# custom packages
using wgf;

Imageh = load("Deblurring/galaxy_blurred_noisy.png");
Imageh = convert(Array{Float64}, Imageh);
pixels = size(Imageh);

# number of iterations
Niter = 500;
# samples from h(y)
M = 20000;
# number of particles
Nparticles = 20000;
# regularisation parameter
lambda = 0.01;
dt = 10^-4;

R = 50;
beta = 3;
x, y = wgf_turbolence(Nparticles, Niter, dt, lambda, Imageh, M, beta, R);

save("turbolence07072020.jld", "x", x, "y", y, "Niter", Niter, "pixels", pixels);

@rimport ks as rks
# grid
Xbins = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
Ybins = range(-0.5 + 1/pixels[1], stop = 0.5 - 1/pixels[1], length = pixels[1]);
gridX = repeat(Xbins, inner=[pixels[1], 1]);
gridY = repeat(Ybins, outer=[pixels[2] 1]);
KDEeval = [gridX gridY];
# KDE
KDEdata = [x[Niter, :] y[Niter, :]];
KDEyWGF = rks.kde(x = KDEdata, var"eval.points" = KDEeval);
deblurringWGF = reshape(rcopy(KDEyWGF[3]), (pixels[1], pixels[2]));
deblurringWGF = map(clamp01nan, deblurringWGF);
# plot
Gray.(reverse(deblurringWGF, dims = 1))
