push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Distributions;
using Statistics;
using StatsBase;
using StatsPlots;
using Interpolations;
using RCall;
@rimport ks as rks

alpha = 4.9^2/3.3^2;
beta = 4.9/3.3.^2;
sigma = sqrt(2log(5.5/5.2));
mu = log(5.2);

# sample from Gamma
x1 = rand(Gamma(alpha, beta), 10^6);
# sample from log-normal
x2 = rand(LogNormal(mu, sigma), 10^6);
# concolution
x = x1 + x2;
# KDE
RKDE = rks.kde(x = x);
KDEy =  abs.(rcopy(RKDE[3]));
KDEx = rcopy(RKDE[2]);

# linear interpolation
itp = interpolate(KDEy, BSpline(Linear()));
s_itp = scale(itp, KDEx);
v = itp(28.38);

plot(KDEx, KDEy)
scatter!([28.38], [v], color = "red")
