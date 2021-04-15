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
using wgf_prior;
using samplers;
include("ct_kde.jl")
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

# sinogram
A = fp_op_parallel2d_line(proj_geom, pixels, pixels);
sinogram = A * vec(CTscan);
sinogram = reshape(Array(sinogram), (:, offsets));
q = filter_proj(sinogram);
sinogram = sinogram./maximum(sinogram);
Gray.(sinogram)

# filtered back projection
fbp = A' * vec(q) .* (pi / nphi);
fbp_img = reshape(fbp, size(CTscan));
Gray.(fbp_img./maximum(fbp_img))

# grid
X1bins = range(-1 + 1/pixels, stop = 1 - 1/pixels, length = pixels);
X2bins = range(-1 + 1/pixels, stop = 1 - 1/pixels, length = pixels);


# functional approximation
# function psi(t)
#     piSample = [transpose(t[1:Nparticles]); transpose(t[(Nparticles+1):(2Nparticles)])];
#     loglik = zeros(size(sinogram));
#     for i=1:nphi
#         for j=1:length(xi)
#         loglik[i, j] = mean(pdf.(Normal.(0, sigma), piSample[1, :] * cos(phi_angle[i]) .+
#             piSample[2, :] * sin(phi_angle[i]) .- xi[j]));
#         end
#     end
#     loglik = -log.(loglik);
#     kl = (phi_angle[2] - phi_angle[1])*(xi[2]-xi[1])*sum(loglik);
#     prior = pdf(MvNormal(m0, Diagonal(sigma0)), piSample);
#     piKDE = kde!(piSample);
#     pihat = piKDE(piSample);
#     kl_prior = mean(log.(pihat./prior));
#     return kl+alpha*kl_prior;
# end

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
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
