push!(LOAD_PATH, "C:/Users/Francesca/Desktop/WGF/myModules")
# Julia packages
using Revise;
using StatsPlots;
using Distributions;
using Statistics;
using StatsBase;
using Random;
using Distances;
using XLSX;
using RCall;
@rimport ks as rks;
# custom packages
using wgf;

# set seed
Random.seed!(1234);

# R example
R"""
library(tictoc)
library(fDKDE)
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

# function computing KDE
function phi(t)
    RKDE = rks.kde(x = t, var"eval.points" = @rget xx);
    return abs.(rcopy(RKDE[3]));
end
# function computing E
function psi(t)
    # entropy
    function remove_non_finite(x)
	       return isfinite(x) ? x : 0
    end
    ent = -mean(remove_non_finite.(t .* log.(t)));
    # kl
    trueH = @rget muKDE;
    refY = @rget xx;
    # approximated value
    delta = refY[2] - refY[1];
    hatH = zeros(1, length(refY));
    # convolution with approximated f
    # this gives the approximated value
    for i=1:length(refY)
        hatH[i] = delta*sum(pdf.(Laplace.(refY, sigU), refY[i]).*t);
    end
    kl = kl_divergence(trueH, hatH);
    return kl-alpha*ent;
end

# get sample from μ
muSample = @rget W;
# get parameter for K
sigU = @rget sigU;

# parameters for WGF
a = 0.5;
alpha = 0.1;
Nparticles = 1000;
dt = 1e-3;
Niter = 10000;
M = 1000;
x0 = sample(muSample, Nparticles, replace = true);
tWGF = @elapsed begin
x = wgf_DKDE_tamed(Nparticles, dt, Niter, alpha, x0, muSample, M, a, sigU);
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
    data <- data.frame(x = x, y = c(tdensity, outcome$DKDE_nonrescaledPI, outcome$DKDE_rescaledCV, outcome$naive_KDE, KDE_wgf$estimate), g = factor(g))
    p2 <- ggplot(data, aes(x, y, color = g)) +
    geom_line(size = 1) +
    scale_color_manual(values = 1:5, labels=c("true f", "fdec, hPI", "fdec, hCV", "naive estimator, hNR", "WGF")) +
    theme(axis.title=element_blank(), text = element_text(size=20), legend.title=element_blank(), aspect.ratio = 2/3)
    # ggsave("sucrase.eps", p2,  height=5)
"""
