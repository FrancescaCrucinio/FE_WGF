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
tEM[:, 1] = readf1[:, 1];
iseEM[:, 1] = readf1[:, 2];
tWGF[:, 1] = readf1[:, 3];
iseWGF[:, 1] = readf1[:, 4];
# d = 2
readf2 = readdlm("mixture_hd/em_vs_wgf_2d.txt", ',', Float64);
tEM[:, 2] = readf2[:, 1];
iseEM[:, 2] = readf2[:, 2];
tWGF[:, 2] = readf2[:, 3];
iseWGF[:, 2] = readf2[:, 4];
# d = 3
readf3 = readdlm("mixture_hd/em_vs_wgf_3d.txt", ',', Float64);
tEM[:, 3] = readf3[:, 1];
iseEM[:, 3] = readf3[:, 2];
tWGF[:, 3] = readf3[:, 3];
iseWGF[:, 3] = readf3[:, 4];
# d = 4
readf4 = readdlm("mixture_hd/em_vs_wgf_4d.txt", ',', Float64);
tEM[:, 4] = readf[1, :];
iseEM[:, 4] = readf[2, :];
tWGF[:, 4] = readf[3, :];
iseWGF[:, 4] = readf[4, :];
