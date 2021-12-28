push!(LOAD_PATH, "/Users/francescacrucinio/Documents/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using LinearAlgebra;
using DelimitedFiles;

means = [0.3 0.7];
variances = [0.07^2; 0.1^2];
# true mean, variance and probability of quadrant
m = means[1]/3 + 2*means[2]/3;
v = variances[1]/3 + 2*variances[2]/3 + means[1]^2/3 + 2*means[2]^2/3 - m^2;

Nrep = 100;
dims = 5;
tSMC1000 = zeros(Nrep, dims);
tWGF1000 = zeros(Nrep, dims);
mWGF1000 = zeros(Nrep, dims);
mSMC1000 = zeros(Nrep, dims);
vWGF1000 = zeros(Nrep, dims);
vSMC1000 = zeros(Nrep, dims);
pWGF1000 = zeros(Nrep, dims);
pSMC1000 = zeros(Nrep, dims);
ksWGF1000 = zeros(Nrep, dims);
ksSMC1000 = zeros(Nrep, dims);
w1WGF1000 = zeros(Nrep, dims);
w1SMC1000 = zeros(Nrep, dims);
for i in 2:dims
    p = ((cdf(Normal(means[1], variances[1]), 0.5) - cdf(Normal(means[1], variances[1]), 0) +
        2*(cdf(Normal(means[2], variances[2]), 0.5) - cdf(Normal(means[2], variances[2]), 0)))/3)^i;
    readf = readdlm("1000smc_vs_wgf_$i.txt", ',', Float64);
    tSMC1000[:, i] = readf[:, 1];
    mSMC1000[:, i] = readf[:, 2];
    vSMC1000[:, i] = readf[:, 3];
    pSMC1000[:, i] = readf[:, 4]/p;
    ksSMC1000[:, i] = readf[:, 5];
    w1SMC1000[:, i] = readf[:, 6];
    tWGF1000[:, i] = readf[:, 7];
    mWGF1000[:, i] = readf[:, 8];
    vWGF1000[:, i] = readf[:, 9];
    pWGF1000[:, i] = readf[:, 10]/p;
    ksWGF1000[:, i] = readf[:, 11];
    w1WGF1000[:, i] = readf[:, 12];
end
# m,v,p and time vs dims
p1 = plot(1:dims, mean(mSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :bottomright, legendfontsize = 15)
plot!(1:dims, mean(mWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1", ylims = (1e-07, 1e-04))
# savefig(p1, "mixture_hd_means.pdf")

p2 = plot(1:dims, mean(vSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :none, legendfontsize = 10)
plot!(1:dims, mean(vWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1", ylims = (1e-07, 1e-04))
# savefig(p2, "mixture_hd_variances.pdf")

p3 = plot(1:dims, mean(pSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :none, legendfontsize = 10)
plot!(1:dims, mean(pWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1")
# savefig(p3, "mixture_hd_probs.pdf")

p4 = plot(1:dims, mean(tSMC1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :blue,
    line = :dash, tickfontsize = 15, label = "SMC-EMS", legend = :none, legendfontsize = 10)
plot!(1:dims, mean(tWGF1000, dims = 1)[:], yaxis = :log10, lw = 3, color = :red,
    line = :solid, label = "Algo 1")
# savefig(p4, "mixture_hd_times2.pdf")

bp1 = boxplot(mSMC1000/v, yaxis = :log10, legend = :none, bar_width = 0.5, range = 0,
    tickfontsize = 15, color = :gray, ylims = [1e-11, 1e-2])
# savefig(bp1, "mixture_hd_bp1.pdf")
bp2 = boxplot(mWGF1000/v, yaxis = :log10, legend = :none, bar_width = 0.5, range = 0,
    tickfontsize = 15, color = :gray, ylims = [1e-11, 1e-2])
# savefig(bp2, "mixture_hd_bp2.pdf")

boxplot(vSMC1000, yaxis = :log10, legend = :none, bar_width = 0.5, range = 0, tickfontsize = 15)
boxplot(vWGF1000, yaxis = :log10, legend = :none, bar_width = 0.5, range = 0, tickfontsize = 15)

boxplot(pSMC1000, yaxis = :log10, legend = :none, bar_width = 0.5, range = 0, tickfontsize = 15)
boxplot(pWGF1000, yaxis = :log10, legend = :none, bar_width = 0.5, range = 0, tickfontsize = 15)

g = repeat(2:dims, inner = Nrep);
bp = violin(g, w1SMC1000[:, 2:dims][:], group = g, legend = :none, side = :left,
    bar_width = 0.5, range = 0, tickfontsize = 15, color = :blue, fillalpha = 0.5, linecolor = :blue)
violin!(g, w1WGF1000[:, 2:dims][:], group = g, legend = :none, side = :right,
    bar_width = 0.5, range = 0, tickfontsize = 15, color = :red, fillalpha = 0.5, linecolor = :red)
plot!(2:dims, mean(w1SMC1000[:, 2:dims], dims = 1)', color = :blue, lw = 2)
plot!(2:dims, mean(w1WGF1000[:, 2:dims], dims = 1)', color = :red, lw = 2)
scatter!(2:dims, mean(w1SMC1000[:, 2:dims], dims = 1)', color = :blue)
scatter!(2:dims, mean(w1WGF1000[:, 2:dims], dims = 1)', color = :red)
