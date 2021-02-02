push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# push!(LOAD_PATH, "C:/Users/francesca/Documents/GitHub/WGF/myModules")
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

# set seed
Random.seed!(1234);

R"""
library(tictoc)
library(fDKDE)
# set seed
set.seed(1234);
#Noise to signal ratio=varU/varX
NSR=0.2

#Sample size
n=500

#Generate data from a normal mixture
X=rnorm(n,5,.4);
X2=matrix(rnorm(n*n,2,1),nrow=n,ncol=n,byrow=TRUE);

pmix=0.75;
tmp=matrix(runif(n,0,1),nrow=1,ncol=n,byrow=TRUE);
X[which(tmp<pmix)]=X2[which(tmp<pmix)];

#Specify error distribution (normal or Laplace in this case) and generate data from this error distribution
errortype="Lap";
#normal case
if (errortype=="norm")
{sigU=sqrt(NSR*var(X));
U=rnorm(n,0,sigU);
varU=sigU^2;}

#Laplace case
if (errortype=="Lap")
{sigU=sqrt(NSR*var(X)/2);
varU=2*sigU^2;
U=rlap(sigU,1,n);}

#Contaminated data
W=as.vector(X+U);

# DKDE
# grid over support
xx <- seq(-2,8,0.1);
# true density
tdensity <- truedens(xx);
# Delaigle's estimators
tic()
outcome<-fDKDE(W,errortype,NSR,n,varU,sigU);
muKDE <- outcome$naive_KDE;
toc()
names(outcome)<-c('PI_bandwidth','DKDE_nonrescaledPI','DKDE_rescaledPI','CV_bandwidth','DKDE_rescaledCV','normal_bandwidth','naive_KDE')
"""

KDEx = @rget xx;
dx = KDEx[2] - KDEx[1];
# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = @rget xx);
    return abs.(rcopy(RKDE[3]));
end
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    prior = pdf.(Normal(m0, sigma0), KDEx);
    kl_prior = dx*kl_divergence(t, prior);
    # kl
    trueMu = @rget muKDE;
    refY = KDEx;
    # approximated value
    delta = refY[2] - refY[1];
    hatMu = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatMu[i] = dx*sum(pdf.(Laplace.(refY, sigU), refY[i]).*t);
    end
#    hatMu[iszero.(hatMu)] .= eps();
    kl = delta*kl_divergence(trueMu, hatMu);
    return kl+alpha*kl_prior;
end

# get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
# number of particles
Nparticles = 500;
# number of samples from μ to draw at each iteration
M = 500;
# time discretisation
dt = 1e-3;
# number of iterations
Niter = 1000;
# regularisation parameter
alpha = 0.011;
# initial distribution
x0 = sample(muSample, M, replace = true);
# prior mean = mean of μ
m0 = mean(muSample);
sigma0 = std(muSample);

tWGF = @elapsed begin
x = wgf_DKDE_tamed(Nparticles, dt, Niter, alpha, x0, m0, sigma0, muSample, M, 0.5, sigU);
end
println("WGF done, $tWGF")

# check convergence
KDEyWGF = mapslices(phi, x, dims = 2);
EWGF = mapslices(psi, KDEyWGF, dims = 2);
plot(EWGF)

# plot
R"""
    # WGF estimator
    library(ks)
    KDE_wgf <- kde($x[$Niter, ], eval.points = xx);
    library(ggplot2)
    g <- rep(1:5, , each = length(xx));
    x <- rep(xx, times = 5);
    data <- data.frame(x = x, y = c(tdensity, outcome$naive_KDE, outcome$DKDE_nonrescaledPI, outcome$DKDE_rescaledCV, KDE_wgf$estimate), g = factor(g))
    p <- ggplot(data, aes(x, y, color = g)) +
    geom_line(size = 2) +
    scale_color_manual(values = c("black", "gray", "red", "orange", "blue"), labels=c(expression(rho(x)), expression(paste("KDE ", mu)), "fdec-hPI", "fdec-hCV", "WGF")) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("simulated_data.eps", p,  height=4)

    var(tdensity - KDE_wgf$estimate)
    var(tdensity - outcome$DKDE_nonrescaledPI)
    var(tdensity - outcome$naive_KDE)
    var(tdensity - outcome$DKDE_rescaledCV)
"""
