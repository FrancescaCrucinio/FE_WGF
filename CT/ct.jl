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
# custom packages
using samplers;
using wgf_prior;


# set seed
Random.seed!(1234);
# CT scan
CTscan = load("CT/LIDC_IDRI_0683_1_048.jpg");
CTscan = convert(Array{Float64}, Gray.(CTscan));
CTscan = CTscan./maximum(CTscan);
# load data image
sinogram = load("CT/noisy_sinogram.png");
sinogram = convert(Array{Float64}, sinogram);
# number of angles
pixels = size(sinogram, 1);
phi_angle = range(0, stop = 2pi, length = pixels);
# offsets
offsets = size(sinogram, 2);
xi = range(-floor(offsets/2), stop = floor(offsets/2), length = offsets);
xi = xi/maximum(xi);

# grid
X1bins = range(-1 + 1/pixels, stop = 1 - 1/pixels, length = pixels);
X2bins = range(-1 + 1/pixels, stop = 1 - 1/pixels, length = pixels);
gridX1 = repeat(X1bins, inner=[pixels, 1]);
gridX2 = repeat(X2bins, outer=[pixels 1]);
KDEeval = [gridX1 gridX2];

# parameters for WGF
# number of particles
Nparticles = 50000;
# number of samples from μ to draw at each iteration
M = 10000;
# time discretisation
dt = 1e-2;
# number of iterations
Niter = 10;
# regularisation parameter
alpha = 0.0001;
# prior mean
m0 = [0; 0];
sigma0 = [0.2; 0.2];
# initial distribution
x0 = sigma0[1]*randn(2, Nparticles);
# variance of normal describing alignment
sigma = 0.02;

# WGF
tWGF = @elapsed begin
x1, x2, E = wgf_ct_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, M, sinogram, phi_angle, xi, sigma);
end

plot(E[:])


using RCall;
@rimport ks as rks;
KDEyWGFfinal = rks.kde(x = [x1[Niter, :] x2[Niter, :]], var"eval.points" = KDEeval);
KDEyWGFfinal = abs.(rcopy(KDEyWGFfinal[3]));
# ise
petWGF = reshape(KDEyWGFfinal, (pixels, pixels));
petWGF = reverse(petWGF, dims=1);
petWGF = petWGF/maximum(petWGF);
var(petWGF .- CTscan)
Gray.(petWGF)
