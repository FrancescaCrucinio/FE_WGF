module smcems

using Distributions;
using Statistics;
using StatsBase;

export smc_AT_approximated_potential
export optimal_bandwidthESS
export weightedKDE

#=
 SMC for analytically tractable example (approximated potential)
OUTPUTS
1 - particle locations
2 - particle weights
INPUTS
'N' number of particles
'Niter' number of time steps
'epsilon' standard deviation for Gaussian smoothing kernel
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
=#
function smc_AT_approximated_potential(N, Niter, epsilon, x0, M)
    # initialise a matrix x storing the particles
    x = zeros(Niter,N);
    # initialise a matrix W storing the weights
    W = zeros(Niter,N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # uniform weights at time n = 1
    W[1, :] = ones(1, N)/N;

    for n=2:Niter
        # get samples from h(y)
        y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);
        # ESS
        ESS=1/sum(W[n-1,:].^2);
        # RESAMPLING
        if(ESS < N/2)
            indices = trunc.(Int, mult_resample(W[n-1,:], N));
            x[n,:] = x[n-1, indices];
            W[n,:] .= 1/N;
        else
            x[n,:] = x[n-1,:];
            W[n,:] = W[n-1,:];
        end

        # Compute h^N_{n}
        hN = zeros(M,1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # Markov kernel: Random walk step
        x[n, :] = x[n, :] + epsilon*randn(N, 1);

        # update weights
        for i=1:N
            g = pdf.(Normal.(x[n, i], 0.045), y);
            potential = sum(g ./ hN);
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

#=
 Weighted kernel density estimation
OUTPUTS
1 - value of kernel density estimator at KDEx
INPUTS
'x' data
'W' weights
'bw' bandwidth
'KDEx' position at which evaluate the density
=#
function weightedKDE(x, W, bw, KDEx)
    # number of positions at which evaluate the density
    n = length(KDEx);
    # KDE
    KDEy = zeros(n, 1);
    for i=1:n
        # use Gaussian kernel
        KDEy[i] = sum(W .* pdf.(Normal.(x, bw), KDEx[i]));
    end

    return KDEy
end

end
