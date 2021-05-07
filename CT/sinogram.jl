# Julia packages
using Images;
using TomoForward;
using XfromProjections;
using Noise;

# CT scan
CTscan = load("CT/LIDC_IDRI_0683_1_048_128p.jpg");
CTscan = convert(Array{Float64}, Gray.(CTscan));
pixels = size(CTscan, 1);
Gray.(CTscan)

# number of angles
nphi = size(CTscan, 1);
# angles
phi_angle = range(0, stop = 2pi, length = nphi);
# number of offsets
offsets = 185;
proj_geom = ProjGeom(1.0, offsets, phi_angle);
xi = range(-floor(offsets/2), stop = floor(offsets/2), length = offsets);
xi = xi/maximum(xi);

# sinogram
A = fp_op_parallel2d_line(proj_geom, pixels, pixels);
sinogram = A * vec(CTscan);
sinogram = reshape(Array(sinogram), (:, offsets));
noisy_sinogram = poisson(sinogram);
noisy_sinogram_std = noisy_sinogram./maximum(noisy_sinogram);
Gray.(noisy_sinogram_std)
save("sinogram_128p.png", colorview(Gray, noisy_sinogram_std));

# # filtered back projection
# q = filter_proj(sinogram);
# fbp = A' * vec(q) .* (pi / nphi);
# fbp_img = reshape(fbp, size(CTscan));
# Gray.(fbp_img./maximum(fbp_img))
