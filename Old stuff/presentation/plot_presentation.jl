using StatsPlots;
using LaTeXStrings;

pyplot()
f(x) = x.^2;
g(x) = x.^4;
h(x) = -x.^2;
j(x) = -x.^4;

p = StatsPlots.plot(f, -1, 1, lw = 3, label = L"$x^2$", color=1,
    legendfontsize = 15, legend=:top);
StatsPlots.plot!(g, -1, 1, lw = 3, label = L"$x^4$", color=2);
StatsPlots.plot!(h, -1, 1, lw = 3, label = L"$-x^2$", color=4);
StatsPlots.plot!(j, -1, 1, lw = 3, label = L"$-x^4$", color=5);

savefig(p, "lambda_convexity.pdf")
