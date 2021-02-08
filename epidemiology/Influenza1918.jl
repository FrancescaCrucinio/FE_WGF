push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using RCall;
@rimport ks as rks;
# custom packages
using wgf_prior;
include("RL.jl")
# set seed
Random.seed!(1234);

# fitted Gaussian approximating K
K(x, y) = 0.595*pdf.(Normal(8.63, 2.56), y .- x) +
        0.405*pdf.(Normal(15.24, 5.39), y .- x);
R"""
library(incidental)
library(tictoc)
# death counts
death_counts <- spanish_flu$Philadelphia

# RIDE estimator
tic()
Philadelphia_model <- fit_incidence(
  reported = spanish_flu$Philadelphia,
  delay_dist = spanish_flu_delay_dist$proportion)
toc()
"""
# get counts from μ
muCounts = Int.(@rget death_counts);
# get sample from μ
muSample = vcat(fill.(1:length(muCounts), muCounts)...);
# shuffle sample
shuffle!(muSample);
# x axis = time (122 days)
KDEx = 1:length(muCounts);

# functional approximation
function psi(piSample)
    loglik = zeros(1, length(muSample));
    for i=1:length(muSample)
        loglik[i] = mean(K.(piSample, muSample[i]));
    end
    loglik = -log.(loglik);
    kl = mean(loglik);
    prior = pdf.(Normal(m0, sigma0), piSample);
    Rpihat = rks.kde(x = piSample, var"eval.points" = piSample);
    pihat = abs.(rcopy(Rpihat[3]));
    kl_prior = mean(log.(pihat./prior));
    return kl+alpha*kl_prior;
end

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-1;
# number of iterations
Niter = 3000;
# initial distribution
x0 = sample(muSample, M, replace = true) .- 9;
# prior mean = mean of μ shifted back by 9 days
m0 = mean(muSample) - 9;
sigma0 = std(muSample);
# regularisation parameter
alpha = 0.01;
runtimeWGF = @elapsed begin
# run WGF
x = wgf_flu_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M, 0.5);
end
RKDEyWGF = rks.kde(x = x[Niter, :], var"eval.points" = KDEx);
KDEyWGF = abs.(rcopy(RKDEyWGF[3]));
# check convergence
EWGF = mapslices(psi, x, dims = 2);
plot(EWGF)

# RL
# initial distribution
rho0 = [muCounts[9:end]; zeros(8, 1)];
# delay distribution
R"""
K_prop <- spanish_flu_delay_dist$proportion
K_day <- spanish_flu_delay_dist$days
"""
K_prop =@rget K_prop;
K_day = Int.(@rget K_day);

KDisc = eps()*ones(length(muCounts), length(muCounts));
for i=1:length(muCounts)
    for j=1:length(muCounts)
        if (i - j >= 1 && i - j<= length(K_day))
            KDisc[i, j] = K_prop[i - j];
        end
    end
end
runtimeRL = @elapsed begin
rhoCounts = RL(KDisc, muCounts, 100, rho0);
end
# recovolve WGF
refY = KDEx;
delta = refY[2] - refY[1];
KDEyRec = zeros(1, length(refY));
for i=1:length(refY)
    KDEyRec[i] = delta*sum(K.(KDEx, refY[i]).*KDEyWGF);
end

# recovolve RL
RLyRec = zeros(1, length(refY));
for i=1:length(refY)
    t = refY[i] .- KDEx;
    nonnegative = (t .>= 1) .& (t .<= 31);
        RLyRec[i] = delta*sum(K_prop[t[nonnegative]].*rhoCounts[100, nonnegative]);
end
# plot
R"""
library(ggplot2)
g <- rep(1:3, , each = length(spanish_flu$Date));
data <- data.frame(x = rep(spanish_flu$Date, times = 3), y = c($rhoCounts[100,]/sum($rhoCounts[100,]), Philadelphia_model$Ihat/sum(Philadelphia_model$Ihat), $KDEyWGF, g = factor(g))
p1 <- ggplot(data, aes(x, y, color = g)) +
geom_line(size = 2) +
scale_color_manual(values = c("gray", "red", "blue"), labels=c("RL", "RIDE", "WGF")) +
theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3, , legend.position="none")
# ggsave("flu1918_reconstruction.eps", p1,  height=4)

# reconstructed death counts
g <- rep(1:4, , each = length(spanish_flu$Date));
data <- data.frame(x = rep(spanish_flu$Date, times = 4), y = c($RLyRec, Philadelphia_model$reported, Philadelphia_model$Chat, $KDEyRec*sum(Philadelphia_model$reported)), g = factor(g))
p2 <- ggplot(data, aes(x, y, color = g)) +
geom_line(data = data[data$g!=1, ], size = 2) +
geom_point(data = data[data$g==1, ], size = 3, shape = 3) +
scale_color_manual(values = c("black", "gray", "red", "blue"), labels=c("recorded", "RL", "RIDE", "WGF")) +
theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3, legend.position="none")
# ggsave("flu1918_reconv.eps", p2,  height=4)

# save legend separately
p3 <- ggplot(data, aes(x, y, color = g)) +
geom_line(data = data[data$g!=1, ], size = 2) +
geom_point(data = data[data$g==1, ], size = 3, shape = 3) +
scale_color_manual(values = c("black", "gray", "red", "blue"), labels=c("recorded", "RL", "RIDE", "WGF")) +
theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3, legend.position="bottom")

g_legend <- function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}

mylegend <- g_legend(p3)
library(grid)
grid.draw(mylegend)

# ggsave("flu1918_legend.eps", mylegend,  height=2)
"""
