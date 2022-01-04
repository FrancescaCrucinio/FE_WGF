push!(LOAD_PATH, "/Users/francescacrucinio/Documents/WGF/myModules")
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using DelimitedFiles;

Nrep = 100;
dims = 10;
tSMC1000 = zeros(Nrep, dims);
tWGF1000 = zeros(Nrep, dims);
w1WGF1000 = zeros(Nrep, dims);
w1SMC1000 = zeros(Nrep, dims);
for i in 2:dims
    readf = readdlm("1000smc_vs_wgf_$i.txt", ',', Float64);
    tSMC1000[:, i] = readf[:, 1];
    w1SMC1000[:, i] = readf[:, 6];
    tWGF1000[:, i] = readf[:, 7];
    w1WGF1000[:, i] = readf[:, 12];
end

p = plot(2:dims, mean(tSMC1000[:,2:dims], dims = 1)[:], lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, legend = :none)
plot!(2:dims, mean(tWGF1000[:,2:dims], dims = 1)[:], lw = 3, color = :red,
    line = :solid, label = "", legend = :none)
scatter!(2:dims, mean(tSMC1000[:,2:dims], dims = 1)[:], color = :blue, markersize = 6)
scatter!(2:dims, mean(tWGF1000[:,2:dims], dims = 1)[:], color = :red, markersize = 6)
# savefig(p, "mixture_hd_times.pdf")

g = repeat(2:dims, inner = Nrep);
bp = violin(g, w1SMC1000[:, 2:dims][:], group = g, label = "", side = :left,
    bar_width = 0.5, range = 0, tickfontsize = 15, color = :blue, fillalpha = 0.5, linecolor = :blue)
violin!(g, w1WGF1000[:, 2:dims][:], group = g, label = "", side = :right,
    bar_width = 0.5, range = 0, tickfontsize = 15, color = :red, fillalpha = 0.5, linecolor = :red)
plot!(2:dims, mean(w1SMC1000[:, 2:dims], dims = 1)', color = :blue, lw = 2, label = "")
plot!(2:dims, mean(w1WGF1000[:, 2:dims], dims = 1)', color = :red, lw = 2, label = "")
scatter!(2:dims, mean(w1SMC1000[:, 2:dims], dims = 1)', color = :blue, label = "SMC-EMS")
scatter!(2:dims, mean(w1WGF1000[:, 2:dims], dims = 1)', color = :red, label = "Algo 1",
    legendfontsize = 15, legend = :topleft)
# savefig(bp, "mixture_hd_w1.pdf")
