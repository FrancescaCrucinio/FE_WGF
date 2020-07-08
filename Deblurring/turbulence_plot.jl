push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using DelimitedFiles;
using JLD;
using RCall;
using Images;
@rimport ks as rks

# entropy function
function remove_non_finite(x)
       return isfinite(x) ? x : zero(x)
end

# clear image
Imagef = load("Deblurring/galaxy.png");
Imagef = Gray.(Imagef);
Imagef = convert(Array{Float64}, Imagef);

x = load("Deblurring/turbolence05072020.jld", "x");
y = load("Deblurring/turbolence05072020.jld", "y");
pixels = load("Deblurring/turbolence05072020.jld", "pixels");
Niter = load("Deblurring/turbolence05072020.jld", "Niter");

# grid
Xbins = range(-1 + 1/pixels[2], stop = 1 - 1/pixels[2], length = pixels[2]);
Ybins = range(-0.5 + 1/pixels[1], stop = 0.5 - 1/pixels[1], length = pixels[1]);
gridX = repeat(Xbins, inner=[pixels[1], 1]);
gridY = repeat(Ybins, outer=[pixels[2] 1]);
KDEeval = [gridX gridY];
# KDE
KDEdata = [x[Niter, :] y[Niter, :]];
KDEyWGF = rks.kde(x = KDEdata, var"eval.points" = KDEeval);
deblurringWGF = reshape(rcopy(KDEyWGF[3]), (pixels[2], pixels[1]));
deblurringWGF = map(clamp01nan, deblurringWGF);
# plot
Gray.(reverse(deblurringWGF, dims = 1))
save("galaxy_reconsruction.png", Gray.(reverse(deblurringWGF, dims = 1));
# mise
miseWGF = (norm(deblurringWGF - Imagef).^2)/length(deblurringWGF);
# entropy
deblurringWGF_ent = -mean(remove_non_finite.(deblurringWGF .* log.(deblurringWGF)));
