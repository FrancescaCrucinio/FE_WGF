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
tEM = ones(Nrep, dims);
iseEM = ones(Nrep, dims);
tWGF = ones(Nrep, dims);
iseWGF = ones(Nrep, dims);
# d = 1
readf1 = readdlm("mixture_hd/em_vs_wgf_1d.txt", ',', Float64);
tEM[:, 1] = readf1[1:100];
iseEM[:, 1] = readf1[101:200];
tWGF[:, 1] = readf1[201:300];
iseWGF[:, 1] = readf1[301:400];
# d = 2
readf2 = readdlm("mixture_hd/em_vs_wgf_2d.txt", ',', Float64);
tEM[:, 2] = readf2[1:100];
iseEM[:, 2] = readf2[101:200];
tWGF[:, 2] = readf2[201:300];
iseWGF[:, 2] = readf2[301:400];
# d = 3
readf3 = readdlm("mixture_hd/em_vs_wgf_3d.txt", ',', Float64);
tEM[:, 3] = readf3[1:100];
iseEM[:, 3] = readf3[101:200];
tWGF[:, 3] = readf3[201:300];
iseWGF[:, 3] = readf3[301:400];
# # d = 4
readf4 = readdlm("mixture_hd/em_vs_wgf_4d.txt", ',', Float64);
tEM[:, 4] = readf4[1:100];
iseEM[:, 4] = readf4[101:200];
tWGF[:, 4] = readf4[201:300];
iseWGF[:, 4] = readf4[301:400];

gain = iseEM./iseWGF;

bp = boxplot(gain, yaxis = :log10, legend = :none, bar_width = 0.5, range = 0, tickfontsize = 15, color = :gray)
# savefig(bp, "mixture_hd_boxplot.pdf")
p = plot(1:dims, mean(tEM, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "OSL-EM", legend = :bottomright, legendfontsize = 10)
plot!(p, 1:dims, mean(tWGF, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1")
# savefig(p, "mixture_hd_times.pdf")
