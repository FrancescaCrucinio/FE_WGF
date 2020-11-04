push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using KernelEstimator;
using Random;
using ImageMagick;
using TestImages, Colors;
using Images;
using DelimitedFiles;
# custom packages
using diagnostics;
using smcems;
using wgf;
using samplers;

Imagef = load("Deblurring/BC.png");
Imagef = Gray.(Imagef);
Imagef
Imagef = convert(Array{Float64}, Imagef);
pixels = size(Imagef);

function acos_domain(x)
       return abs(x)<=1 ? acos(x) : zero(x)
end

sigma = 0.02;
# create empty image
Imageh = zeros(pixels);
# set coordinate system over image
# x is in [-1, 1]
Xbins = range(-1+ 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
# y is in [-0.5, 0.5]
Ybins = range(0.5 - 1/pixels[1], stop = -0.5 + 1/pixels[1], length = pixels[1]);
# build grid with this coordinates
gridX = repeat(transpose(Xbins), outer=[pixels[1] 1]);
gridY = repeat(Ybins, outer=[1 pixels[2]]);

Threads.@threads for i=1:pixels[2]
    # get (u, v)
    u = Xbins[i];
    prec = acos_domain.(u .- gridX);
    @simd for j=1:pixels[1]
        v = Ybins[j];
        Imageh[j, i] = sum(Imagef .* prec.^(-0.5) .*
            pdf.(Normal(0, sigma), prec .- asin.(v .- gridY)));
    end
end

# normalize image
Imageh =  Gray.(Imageh);
Imageh
# save("Blurred_image.png", Imageh);
