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
dims = 6;
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
p1 = plot(1:dims, mean(mSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :bottomright, legendfontsize = 10)
plot!(1:dims, mean(mWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "WGF")
# savefig(p1, "mixture_hd_means.pdf")

p2 = plot(1:dims, mean(vSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :bottomright, legendfontsize = 10)
plot!(1:dims, mean(vWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "WGF")
# savefig(p1, "mixture_hd_variances.pdf")

p3 = plot(1:dims, mean(pSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :bottomright, legendfontsize = 10)
plot!(1:dims, mean(pWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "WGF")
# savefig(p3, "mixture_hd_probs.pdf")

p4 = plot(1:dims, mean(tSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :bottomright, legendfontsize = 10)
plot!(1:dims, mean(tWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "WGF")
# savefig(p4, "mixture_hd_times.pdf")
