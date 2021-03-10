push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Distributions;
using Statistics;
using StatsBase;
using Random;
using JLD;
using RCall;
R"library(ggplot2)"
Nparticles = load("analytically tractable/comparison_uniform17062020.jld", "Nparticles");
tSMCuniform = load("analytically tractable/comparison_uniform17062020.jld", "tSMC");
tWGFuniform = load("analytically tractable/comparison_uniform17062020.jld", "tWGF");
diagnosticsSMCuniform = load("analytically tractable/comparison_uniform17062020.jld", "diagnosticsSMC");
diagnosticsWGFuniform = load("analytically tractable/comparison_uniform17062020.jld", "diagnosticsWGF");
# qSMC = load("analytically tractable/comparison_uniform17062020.jld", "qdistSMC");
# qWGF = load("analytically tractable/comparison_uniform17062020.jld", "qdistWGF");

tSMCdelta = load("analytically tractable/comparison_delta18062020.jld", "tSMC");
tWGFdelta = load("analytically tractable/comparison_delta18062020.jld", "tWGF");
diagnosticsSMCdelta = load("analytically tractable/comparison_delta18062020.jld", "diagnosticsSMC");
diagnosticsWGFdelta = load("analytically tractable/comparison_delta18062020.jld", "diagnosticsWGF");
qSMCdelta = load("analytically tractable/comparison_delta18062020.jld", "qdistSMC");
qWGFdelta = load("analytically tractable/comparison_delta18062020.jld", "qdistWGF");




markers = [:circle :rect :diamond :xcross :star5];
labels = ["N=100" "N=500" "N=1000" "N=5000" "N=10000"];
pyplot()
p = plot([tSMCuniform tWGFuniform tWGFdelta],
    [diagnosticsSMCuniform[:, 3] diagnosticsWGFuniform[:, 3] diagnosticsWGFdelta[:, 3]],
    lw = 3, label = [L"SMC - Unif" L"WGF - Unif" L"WGF - $\delta$"],
    xaxis=:log, color = [1 2 3], ylims = (0, 1.5), legend=:outerright);
for i=1:length(tSMCuniform)
    scatter!(p, [tSMCuniform[i]], [diagnosticsSMCuniform[i, 3]],
        markersize = 9, label = labels[i], color = :black, marker = markers[i],
        markerstrokewidth=0);
    scatter!(p, [tSMCuniform[i] tWGFuniform[i] tWGFdelta[i]],
        [diagnosticsSMCuniform[i, 3] diagnosticsWGFuniform[i, 3] diagnosticsWGFdelta[i, 3]],
        markersize = 9, label = ["" ""], color = [1 2 3], marker = markers[i],
        markerstrokewidth=0);
end

runtime = [tSMCuniform tWGFuniform tWGFdelta];
mise = [diagnosticsSMCuniform[:, 3] diagnosticsWGFuniform[:, 3] diagnosticsWGFdelta[:, 3]];
N = repeat([100; 500; 1000; 5000; 10000], outer=[3, 1]);
group = repeat([1; 2; 3], inner=[5, 1]);
R"""
    data <- data.frame(x = c($runtime), y = c($mise), z = $N, g = $group);
    ggplot(data, aes(x, y, group = factor(g), color = factor(g))) +
    geom_line() +
    geom_point(aes(shape = factor(z)))
"""
