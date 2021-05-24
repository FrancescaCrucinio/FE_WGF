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
dims = 4;
tEM = zeros(Nrep, dims);
iseEM = zeros(Nrep, dims);
tWGF = zeros(Nrep, dims);
iseWGF = zeros(Nrep, dims);
# d = 1
readf1 = readdlm("mixture_hd/em_vs_wgf_1d.txt", ',', Float64);
tEM[:, 1] = readf[1, :];
iseEM[:, 1] = readf[2, :];
tWGF[:, 1] = readf[3, :];
iseWGF[:, 1] = readf[4, :];
# d = 2
readf1 = readdlm("mixture_hd/em_vs_wgf_2d.txt", ',', Float64);
tEM[:, 2] = readf[1, :];
iseEM[:, 2] = readf[2, :];
tWGF[:, 2] = readf[3, :];
iseWGF[:, 2] = readf[4, :];
# d = 3
readf1 = readdlm("mixture_hd/em_vs_wgf_3d.txt", ',', Float64);
tEM[:, 3] = readf[1, :];
iseEM[:, 3] = readf[2, :];
tWGF[:, 3] = readf[3, :];
iseWGF[:, 3] = readf[4, :];
# d = 4
readf1 = readdlm("mixture_hd/em_vs_wgf_4d.txt", ',', Float64);
tEM[:, 4] = readf[1, :];
iseEM[:, 4] = readf[2, :];
tWGF[:, 4] = readf[3, :];
iseWGF[:, 4] = readf[4, :];



plot(alpha,  mean(E, dims = 2))
