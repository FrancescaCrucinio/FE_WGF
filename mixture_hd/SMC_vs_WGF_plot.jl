push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using KernelEstimator;
using DelimitedFiles;

Nrep = 100;
dims = 3;
tSMC1000 = ones(Nrep, dims);
tWGF1000 = ones(Nrep, dims);
entWGF1000 = ones(Nrep, dims);
entSMC1000 = ones(Nrep, dims);
mWGF1000 = ones(Nrep, dims);
mSMC1000 = ones(Nrep, dims);
vWGF1000 = ones(Nrep, dims);
vSMC1000 = ones(Nrep, dims);
pWGF1000 = ones(Nrep, dims);
pSMC1000 = ones(Nrep, dims);
for i in 1:dims
    readf = readdlm("mixture_hd/1000smc_vs_wgf_$i.txt", ',', Float64);
    tSMC1000[:, i] = readf[:, 1];
    entSMC1000[:, i] = readf[:, 5];
    mSMC1000[:, i] = readf[:, 2];
    vSMC1000[:, i] = readf[:, 3];
    pSMC1000[:, i] = readf[:, 4];
    tWGF1000[:, i] = readf[:, 6];
    entWGF1000[:, i] = readf[:, 10];
    mWGF1000[:, i] = readf[:, 7];
    vWGF1000[:, i] = readf[:, 8];
    pWGF1000[:, i] = readf[:, 9];
end
# m,v,p and time vs dims
p1 = plot(1:dims, mean(mSMC1000, dims = 1)[:])
plot!(1:dims, mean(mWGF1000, dims = 1)[:])

p2 = plot(1:dims, mean(vSMC1000, dims = 1)[:])
plot!(1:dims, mean(vWGF1000, dims = 1)[:])

p3 = plot(1:dims, mean(pSMC1000, dims = 1)[:])
plot!(1:dims, mean(pWGF1000, dims = 1)[:])

p4 = plot(1:dims, mean(tSMC1000, dims = 1)[:])
plot!(1:dims, mean(tWGF1000, dims = 1)[:])

p5 = plot(1:dims, mean(entSMC1000, dims = 1)[:])
plot!(1:dims, mean(entWGF1000, dims = 1)[:])
