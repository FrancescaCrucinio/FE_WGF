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
using TomoForward;
using XfromProjections;
# custom packages
using samplers;
using wgf_prior;


# set seed
Random.seed!(1234);

# CT scan
CTscan = load("CT/LIDC_IDRI_0683_1_048.jpg");
CTscan = convert(Array{Float64}, CTscan);
pixels = size(CTscan, 1);
Gray.(CTscan)

# number of angles
nphi = size(CTscan, 1);
# angles
phi_angle = range(0, stop = 2pi, length = nphi);
# number of offsets
offsets = 729;
proj_geom = ProjGeom(1.0, offsets, phi_angle);
xi = range(-floor(offsets/2), stop = floor(offsets/2), length = offsets);
xi = xi/maximum(xi);
#
# # sinogram
# A = fp_op_parallel2d_line(proj_geom, pixels, pixels);
# sinogram = A * vec(CTscan);
# sinogram = reshape(Array(sinogram), (:, offsets));
# sinogram = sinogram./maximum(sinogram);
# save("sinogram.png", colorview(Gray, sinogram));

sinogram = load("CT/sinogram.png");
sinogram = convert(Array{Float64}, sinogram);
# # filtered back projection
# q = filter_proj(sinogram);
# fbp = A' * vec(q) .* (pi / nphi);
# fbp_img = reshape(fbp, size(CTscan));
# Gray.(fbp_img./maximum(fbp_img))

# grid
X1bins = range(-1 + 1/pixels, stop = 1 - 1/pixels, length = pixels);
X2bins = range(-1 + 1/pixels, stop = 1 - 1/pixels, length = pixels);

# parameters for WGF
# number of particles
Nparticles = 1000;
# number of samples from μ to draw at each iteration
M = 1000;
# time discretisation
dt = 1e-2;
# number of iterations
Niter = 50;
# regularisation parameter
alpha = 0.0017;
# prior mean
m0 = [0; 0];
sigma0 = [0.2; 0.2];
# initial distribution
x0 = sigma0[1]*randn(2, Nparticles);
#x0 = rand(2, Nparticles);
# variance of normal describing alignment
sigma = 0.02;

# WGF
tWGF = @elapsed begin
x1, x2, E = wgf_ct_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, M, sinogram, phi_angle, xi, sigma);
end

piKDE = kde([x1[Niter, :] x2[Niter, :]]);
res = pdf(piKDE, X1bins, X2bins);
Gray.(res./maximum(res))

plot(E[:])
