using PolynomialRoots;
using StatsPlots;
using Distributions

sigmaK = 0.45^2;
sigmaPi = 0.43^2;
sigmaMu = sigmaPi + sigmaK;
sigma0 = 0.1^2;

alpha = range(0, stop = 1, length = 100);
E = zeros(length(alpha));
beta = zeros(length(alpha));
for i = 1:length(alpha)
    a = alpha[i];
    b = 2*alpha[i]*sigmaK + (1 - alpha[i])*sigma0;
    c = -sigmaMu*sigma0 + sigma0*sigmaK + alpha[i]*sigmaK^2 - 2*alpha[i]*sigmaK;
    d = -alpha[i]*sigma0*sigmaK^4;
    r = roots([d, c, b, a]);
    beta_index = [(isreal(r[j]) && real(r[j]) > 0) for j=1:length(r)];
    beta[i] = real(r[beta_index])[1];
    E[i] = 0.5*log(2*pi*(beta[i]+sigmaK)) + 0.5*sigmaMu/(beta[i]+sigmaK) + 0.5*alpha[i]*
        (log(sigma0/beta[i]) + beta[i]/sigma0 -1);
end

p1 = plot(alpha, beta, lw = 3, tickfontsize = 15, legend = :none)
hline!(p1, [sigmaPi], lw = 3)
# savefig(p1, "AT_beta.pdf")
p2 = plot(alpha, E, yaxis = :log10, lw = 3, tickfontsize = 15, legend = :none)
hline!(p2, [E[1]], lw = 3)
# savefig(p2, "AT_functional.pdf")

KDEx = range(-3, stop = 3, length = 100);
p3 = plot(KDEx, pdf.(Normal(0, sqrt(sigmaPi)), KDEx), lw = 4, color = :black,
    tickfontsize = 15, label = "pi")
plot!(p3, KDEx, pdf.(Normal(0, sqrt(beta[end])), KDEx), lw = 3, color = 1,
    tickfontsize = 15, label = "posterior", line = :dash)
alpha_cv = 4e-05;
a = alpha_cv;
b = 2*alpha_cv*sigmaK + (1 - alpha_cv)*sigma0;
c = -sigmaMu*sigma0 + sigma0*sigmaK + alpha_cv*sigmaK^2 - 2*alpha_cv*sigmaK;
d = -alpha_cv*sigma0*sigmaK^4;
r = roots([d, c, b, a]);
beta_index = [(isreal(r[j]) && real(r[j]) > 0) for j=1:length(r)];
beta_cv = real(r[beta_index])[1];
plot!(p3, KDEx, pdf.(Normal(0, sqrt(beta_cv)), KDEx), lw = 3, color = 2,
    tickfontsize = 15, label = "cross-validation", legendfontsize = 10, line = :dash)
# savefig(p3, "AT_prior.pdf")
