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
dt = 10^-3;

R = 50;
beta = 3;
x, y = wgf_turbolence(Nparticles, Niter, dt, lambda, Imageh, M, beta, R);

Xbins = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
Ybins = range(0.5 - 1/pixels[1], stop = -0.5 + 1/pixels[1], length = pixels[1]);
KDEyWGF =  KernelDensity.kde((y[end, :], x[end, :]));
resWGF = pdf(KDEyWGF, Ybins, Xbins);

# normalize image
resWGF = map(clamp01nan, resWGF);


save("resWGF10-3.jld", "resWGF", resWGF)

Gray.(resWGF)
Gray.(Imageh)

Imagef = load("Deblurring/galaxy.png");
Imagef = Gray.(Imagef);
Imagef
Imagef = convert(Array{Float64}, Imagef);

d = Euclidean()
d(Gray.(resWGF), Gray.(Imagef))

(norm(resWGF - Imagef).^2)/length(resWGF)
save("galaxy_reconstruction.png", Gray.(resWGF));
