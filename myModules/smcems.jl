module smcems

using Distributions;
using Statistics;
using StatsBase;

using samplers;

export smc_gaussian_mixture
export smc_AT_approximated_potential
export optimal_bandwidthESS
export weightedKDE
export AT_alpha_SMC


#= SMC for gaussian mixture exmaple
OUTPUTS
1 - particle locations
2 - particle weights
INPUTS
'N' number of particles
'Niter' number of time steps
'epsilon' standard deviation for Gaussian smoothing kernel
'x0' initial distribution.
user selected initial distribution
'muSample' sample from μ(y)
'M' number of samples from μ(y) to be drawn at each iteration
=#
function smc_gaussian_mixture(N, Niter, epsilon, x0, muSample, M)
    # initialise a matrix x storing the particles at each time step
    x = zeros(Niter,N);
    # initialise a matrix W storing the weights at each time step
    W = zeros(Niter,N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # uniform weights at time n = 1
    W[1, :] = ones(1, N)/N;

    for n=2:Niter
        # samples from h(y)
        y = sample(muSample, M, replace = true);
        # ESS
        # ESS=1/sum(W[n-1,:].^2);
        # RESAMPLING
        # if(ESS < N/2)
            indices = trunc.(Int, mult_resample(W[n-1,:], N));
            x[n,:] = x[n-1, indices];
            W[n,:] .= 1/N;
        # else
        #    x[n,:] = x[n-1,:];
        #    W[n,:] = W[n-1,:];
        # end

        # Compute h^N_{n}
        hN = zeros(M,1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # Markov kernel: Random walk step
        x[n, :] = x[n, :] + epsilon*randn(N, 1);

        # update weights
        for i=1:N
            K = pdf.(Normal.(x[n, i], 0.045), y);
            potential = sum(K ./ hN);
            # update weight
            W[n, i] = potential;
        end
        # normalise weights
        W[n, :] = W[n, :] ./ sum(W[n, :]);
    end
    return x, W
end

#= Multinomial resampling
OUTPUTS
1- indices of resampled particles
INPUTS
'W' vector of weights
'N' number of particles to sample (in most cases N = length(W))
=#
function mult_resample(W, N)
    # vector to store number of offsprings
    indices = zeros(N, 1);

    # start inverse transfor method
    s = W[1];
    u = sort(rand(N, 1), dims = 1);
    j = 1;
    for i=1:N
    while(s < u[i])
      j = j+1;
      s = s + W[j];
    end
    indices[i] = j;
    end

    return indices
end

#=
 Optimal bandwidth for Gaussian kernel with effective sample size (ESS)
OUTPUTS
1 - bandwidth
INPUTS
'x' particle locations
'W' particle weights
=#
function optimal_bandwidthESS(x, W)
    # ESS
    ESS = sum(W.^2);
    # standard deviation (weighted)
    s = Statistics.std(x, weights(W), corrected = false);
    # bandwidth
    bw = 1.06*s*ESS^(1/5);

    return bw
end
end

# #=
#  SMC for analytically tractable example (approximated potential)
# OUTPUTS
# 1 - particle locations
# 2 - particle weights
# INPUTS
# 'N' number of particles
# 'Niter' number of time steps
# 'epsilon' standard deviation for Gaussian smoothing kernel
# 'x0' user selected initial distribution
# 'M' number of samples from h(y) to be drawn at each iteration
# =#
# function smc_AT_approximated_potential(N, Niter, epsilon, x0, M)
#     # initialise a matrix x storing the particles
#     x = zeros(Niter,N);
#     # initialise a matrix W storing the weights
#     W = zeros(Niter,N);
#     # initial distribution is given as input:
#     x[1, :] = x0;
#     # uniform weights at time n = 1
#     W[1, :] = ones(1, N)/N;
#
#     for n=2:Niter
#         # get samples from h(y)
#         y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);
#         # ESS
#         ESS=1/sum(W[n-1,:].^2);
#         # RESAMPLING
#         if(ESS < N/2)
#             indices = trunc.(Int, mult_resample(W[n-1,:], N));
#             x[n,:] = x[n-1, indices];
#             W[n,:] .= 1/N;
#         else
#             x[n,:] = x[n-1,:];
#             W[n,:] = W[n-1,:];
#         end
#
#         # Compute h^N_{n}
#         hN = zeros(M,1);
#         for j=1:M
#             hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
#         end
#         # Markov kernel: Random walk step
#         x[n, :] = x[n, :] + epsilon*randn(N, 1);
#
#         # update weights
#         for i=1:N
#             g = pdf.(Normal.(x[n, i], 0.045), y);
#             potential = sum(g ./ hN);
#             # update weight
#             W[n, i] = potential;
#         end
#         # normalise weights
#         W[n, :] = W[n, :] ./ sum(W[n, :]);
#     end
#
#     return x, W
# end

#
# #=
#  Weighted kernel density estimation
# OUTPUTS
# 1 - value of kernel density estimator at KDEx
# INPUTS
# 'x' data
# 'W' weights
# 'bw' bandwidth
# 'KDEx' position at which evaluate the density
# =#
# function weightedKDE(x, W, bw, KDEx)
#     # number of positions at which evaluate the density
#     n = length(KDEx);
#     # KDE
#     KDEy = zeros(n, 1);
#     for i=1:n
#         # use Gaussian kernel
#         KDEy[i] = sum(W .* pdf.(Normal.(x, bw), KDEx[i]));
#     end
#
#     return KDEy
# end
#
# #= For entropy computation - removes non finite entries =#
# function remove_non_finite(x)
#        return isfinite(x) ? x : zero(x)
# end
#
# #= Find α giving a target entropy for WGF
# OUTPUTS
# 1 - alpha
# INPUTS
# 'target_entropy' target entropy for the solution
# 'interval' domain of α
# 'threshold' stopping rule
# 'Niter' number of time steps
# 'x0' user selected initial distribution
# 'M' number of samples from h(y) to be drawn at each iteration
# =#
# function AT_alpha_SMC(target_entropy, interval, threshold, Niter, Nparticles, initial_distribution, M)
#     # values at which evaluate KDE
#     KDEx = range(-0, stop = 1, length = 1000);
#     # upper and lower bound for α
#     liminf = interval[1];
#     limsup = interval[2];
#
#     delta_entropy = Inf;
#     epsilon = (limsup + liminf)/2;
#     Nrep = 10;
#     j=1;
#     while (abs(delta_entropy)>threshold && j<=50)
#         actual_entropy  = zeros(Nrep, 1);
#         Threads.@threads for i=1:Nrep
#         if (initial_distribution == "delta")
#             x0 = rand(1)*ones(1, Nparticles);
#         else
#             x0 = rand(1, Nparticles);
#         end
#         ### SMC
#         xSMC, W = smc_AT_approximated_potential(Nparticles, Niter, epsilon, x0, M);
#         # kde
#         bw = sqrt(epsilon^2 + optimal_bandwidthESS(xSMC[Niter, :], W[Niter, :])^2);
#         KDEySMC = weightedKDE(xSMC[end, :], W[end, :], bw, KDEx);
#         actual_entropy[i] = -mean(remove_non_finite.(KDEySMC .* log.(KDEySMC)));
#         end
#         actual_entropy =  mean(actual_entropy);
#         delta_entropy = actual_entropy - target_entropy;
#         if (delta_entropy > 0)
#             limsup = (limsup + liminf)/2;
#         else
#             liminf = (limsup + liminf)/2;
#         end
#         epsilon = (limsup + liminf)/2;
#         println("$j")
#         println("$limsup , $liminf")
#         println("$actual_entropy")
#         println("$delta_entropy")
#         j=j+1;
#     end
#     return limsup, liminf
# end
