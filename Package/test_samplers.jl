push!(LOAD_PATH, "C:/Users/Francesca/OneDrive/Desktop/WGF/Package")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using DelimitedFiles;
# custom modules
using samplers;

# Test 1 - univariate normal
h = StatsBase.fit(Histogram, randn(1000000), nbins=1000);
samples = histogram_sampler(h, 10000);
plot(Normal(0, 1), lw=3)
histogram!(samples, normalize=:pdf)
samples = sample(randn(1000000), 1000, replace=true);

# Test 2 - gamma
h = StatsBase.fit(Histogram, rand(Gamma(3,5),1000000), nbins=1000);
samples = histogram_sampler(h, 10000);
plot(Gamma(3,5), lw=3)
histogram!(samples, normalize=:pdf)

# Test 3 - multivariate normal
s = rand(MvNormal(2, 1), 1000000);
h = StatsBase.fit(Histogram, (s[1, :], s[2, :]), nbins=1000);
samples = histogram_sampler(h, 100000);
histogram2d(samples[:, 2], samples[:, 1])

# Test 4 - PET sinogram
image = readdlm("PET/sinogram.txt", ',', Float64);
support =  [-1.0  1.0;  0.0  2pi];
h = image2histogram(image, support);
samples = histogram_sampler(h, 100000);
histogram2d(samples[:, 2], samples[:, 1])
