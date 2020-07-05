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

save("turbolence05072020.jld", "x", x, "y", y, "Niter", Niter, "pixels", pixels);
