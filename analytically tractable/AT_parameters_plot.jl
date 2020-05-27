# plot for parameter α selection
# Julia packages
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using JLD;
using LaTeXStrings;


lambda = load("analytically tractable/parameters1000.jld", "lambda");
diagnosticsWGF1000 = load("analytically tractable/parameters1000.jld", "diagnosticsWGF");
diagnosticsWGF500 = load("analytically tractable/parameters500.jld", "diagnosticsWGF");

labels = ["N=500" "N=1000"];
pyplot()
p1 = plot(lambda, [diagnosticsWGF500[:, 1] diagnosticsWGF1000[:, 1]], lw = 3, label = labels,
        xlabel=L"$\alpha$", ylabel=L"$\hat{m}_t$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
hline!([0.5], label="True mean");
p2 = plot(lambda, [diagnosticsWGF500[:, 2] diagnosticsWGF1000[:, 2]], lw = 3, label = labels,
        xlabel=L"$\alpha$", ylabel=L"$\hat{\sigma}_t$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
hline!([0.043^2], label =  "True variance");
p3 = plot(lambda, [diagnosticsWGF500[:, 3] diagnosticsWGF1000[:, 3]], lw = 3, label = labels,
        xlabel=L"$\alpha$", ylabel=L"$MSE_{95}$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p4 = plot(lambda, [diagnosticsWGF500[:, 4] diagnosticsWGF1000[:, 4]], lw = 3, label = labels,
        xlabel=L"$\alpha$", ylabel=L"MISE", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p5 = plot(lambda, [diagnosticsWGF500[:, 5] diagnosticsWGF1000[:, 5]], lw = 3, label = labels,
        xlabel=L"$\alpha$", ylabel=L"$E(\rho_t)$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p6 = plot(lambda, [diagnosticsWGF500[:, 6] diagnosticsWGF1000[:, 6]], lw = 3, label = labels,
        xlabel=L"$\alpha$", ylabel=L"$ent(\rho_t)$", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);
p7 = plot(lambda, [diagnosticsWGF500[:, 7] diagnosticsWGF1000[:, 7]], lw = 3, label = labels,
        xlabel=L"$\alpha$", ylabel=L"KL", xguidefontsize=10, yguidefontsize=10, legendfontsize=10);

plot(p1, p2, p3, p4, p5, p6, layout = (2, 3))

savefig(p2, "parameters_var.pdf")
savefig(p4, "parameters_mise.pdf")
savefig(p5, "parameters_E.pdf")
savefig(p6, "parameters_ent.pdf")
