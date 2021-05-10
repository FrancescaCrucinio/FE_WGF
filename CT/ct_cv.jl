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
using Interpolations;
using Images;
using Distances;
using KernelDensity;
using DelimitedFiles;
# custom packages
using samplers;
using wgf_prior;

# set seed
Random.seed!(1234);

# data image
sinogram = load("CT/noisy_sinogram_128p.png");
sinogram = convert(Array{Float64}, sinogram);
# number of angles
pixels = size(sinogram, 1);
# angles
phi_angle = range(0, stop = 2*pi, length = pixels);
# number of offsets
offsets = size(sinogram, 2);
xi = range(-floor(offsets/2), stop = floor(offsets/2), length = offsets);
xi = xi/maximum(xi);


# grid
X1bins = range(-1 + 1/pixels, stop = 1 - 1/pixels, length = pixels);
X2bins = range(-1 + 1/pixels, stop = 1 - 1/pixels, length = pixels);

# parameters for WGF
# number of particles
Nparticles = 50000;
# number of samples from μ to draw at each iteration
M = 10000;
# time discretisation
dt = 1e-2;
# number of iterations
Niter = 10;
# variance of normal describing alignment
sigma = 0.02;
# prior mean
m0 = [0; 0];
sigma0 = [0.2; 0.2];
# regularisation parameter
alpha = range(0.0, stop = 1, length = 2);

# number of repetitions
L = 5;

ent = zeros(length(alpha), L);
kl = zeros(length(alpha), L);
for i=1:length(alpha)
    for l=1:L
        # sample from μ(y)
        muSample = histogram2D_sampler(sinogram, xi, phi_angle, M);
        # initial distribution
        x0 = sigma0[1]*randn(2, Nparticles);
        # WGF
        x1, x2 = wgf_ct_tamed_cv(Nparticles, dt, Niter, alpha[i], x0, m0, sigma0, muSample, sigma);
        # functional
        loglik = zeros(size(sinogram));
        for i=1:length(phi_angle)
            for j=1:length(xi)
                loglik[i, j] = mean(pdf.(Normal.(0, sigma), x1[Niter, :] * cos(phi_angle[i]) .+
                x2[Niter, :] * sin(phi_angle[i]) .- xi[j]));
               end
            end
        loglik = -log.(loglik);
        kl = (phi_angle[2] - phi_angle[1])*(xi[2]-xi[1])*sum(loglik);
        # prior
        prior = pdf(MvNormal(m0, Diagonal(sigma0)), [x1[Niter, :] x2[Niter, :]]');
        pihat = ct_kde([x1[Niter, :] x2[Niter, :]], [x1[Niter, :] x2[Niter, :]]);
        kl_prior = mean(log.(pihat[:]./prior));
        E[i, l] = kl+alpha[i]*kl_prior;
        println("$i, $l")
    end
end

open("resCV.txt", "w") do io
           writedlm(io, [alpha E], ',')
       end
