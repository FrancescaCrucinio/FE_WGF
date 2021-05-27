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
dims = 1;
tSMC = ones(Nrep, dims);
tWGF = ones(Nrep, dims);
entWGF = ones(Nrep, dims);
entSMC = ones(Nrep, dims);
mWGF = ones(Nrep, dims);
mSMC = ones(Nrep, dims);
vWGF = ones(Nrep, dims);
vSMC = ones(Nrep, dims);
pWGF = ones(Nrep, dims);
pSMC = ones(Nrep, dims);
for i in 1:dims
    readf = readdlm("mixture_hd/1000smc_vs_wgf_$i.txt", ',', Float64);
    tSMC[:, i] = readf[:, 1];
    entSMC[:, i] = readf[:, 5];
    mSMC[:, i] = readf[:, 2];
    vSMC[:, i] = readf[:, 3];
    pSMC[:, i] = readf[:, 4];
    tWGF[:, i] = readf[:, 6];
    entWGF[:, i] = readf[:, 10];
    mWGF[:, i] = readf[:, 2];
    vWGF[:, i] = readf[:, 3];
    pWGF[:, i] = readf[:, 4];
end
# m,v,p and time vs dims 
