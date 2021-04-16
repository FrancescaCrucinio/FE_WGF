# Julia packages
using Images;
using TomoForward;
using XfromProjections;

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
# sinogram
A = fp_op_parallel2d_line(proj_geom, pixels, pixels);
sinogram = A * vec(CTscan);
sinogram = reshape(Array(sinogram), (:, offsets));
sinogram = sinogram./maximum(sinogram);
save("sinogram.png", colorview(Gray, sinogram));

# # filtered back projection
# q = filter_proj(sinogram);
# fbp = A' * vec(q) .* (pi / nphi);
# fbp_img = reshape(fbp, size(CTscan));
# Gray.(fbp_img./maximum(fbp_img))
