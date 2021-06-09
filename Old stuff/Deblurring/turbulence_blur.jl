push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using ImageMagick;
using TestImages, Colors;
using Images;
using DelimitedFiles;
using Plots;
using Noise;

Imagef = load("Deblurring/galaxy.png");
Imagef = Gray.(Imagef);
Imagef
Imagef = convert(Array{Float64}, Imagef);
pixels = size(Imagef);

# create empty image
Imageh = zeros(pixels);
# parameters
R = 50/300;
beta = 3;
# PSF
h(r) = (1 .+ r^2/R^2).^(-beta)*beta/(pi*R^2);
plot(h, -1, 1)
# set coordinate system over image
# x is in [-1, 1]
Xbins = range(-1+ 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
# y is in [-0.5, 0.5]
Ybins = range(0.5 - 1/pixels[1], stop = -0.5 + 1/pixels[1], length = pixels[1]);
# build grid with this coordinates
gridX = repeat(transpose(Xbins), outer=[pixels[1] 1]);
gridY = repeat(Ybins, outer=[1 pixels[2]]);

# apply blur
Threads.@threads for i=1:pixels[2]
    # get (u, v)
    u = Xbins[i];
    @simd for j=1:pixels[1]
        v = Ybins[j];
        Imageh[j, i] = sum(Imagef .* (1 .+ ((u .- gridX).^2 .+ (v .- gridY).^2)/R^2).^(-beta))*beta/(pi*R^2);
    end
end

# normalize image
Imageh =  Imageh./maximum(Imageh);
Gray.(Imageh)
Noisyh = add_gauss(Gray.(Imageh));
save("galaxy_blurred.png", Gray.(Imageh));
save("galaxy_blurred_noisy.png", Noisyh);
