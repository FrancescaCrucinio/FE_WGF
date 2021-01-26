module wgf_prior;

using Distributions;
using Statistics;
using LinearAlgebra;

using samplers;

export wgf_prior_gaussian_mixture_tamed
export pda_gaussian_mixture_tamed
export wgf_flu_tamed

#= WGF for gaussian mixture
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'sigma0' standard deviation of prior
'm0' mean of prior
'muSample' sample from μ(y)
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_prior_gaussian_mixture_tamed(N, dt, Niter, alpha, x0, sigma0, m0, muSample, M, a)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;

    for n=1:(Niter-1)
        # samples from h(y)
        y = sample(muSample, M, replace = true);
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
            drift[i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+  dt * (drift./(1 .+ Niter^(-a) * abs.(drift)) .+ alpha*(x[n, :] .- m0)./sigma0^2) .+
            sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x
end

function pda_gaussian_mixture_tamed(N, dt, Niter, alpha, x0, sigma0, m0, muSample, M, a)
    # initialise a matrix x storing the particles
    x = zeros(Niter+1, N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # samples from h(y)
    y = sample(muSample, M, replace = true);
    hN = zeros(M, 1);
    for n=1:Niter
        j = sample(1:M);
        # Compute h^N_{n}
        hN[j] = hN[j] + n/mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        # Langevin steps
        dt = 0.0001/sqrt(n);
        xtilde = x[n, :];
        for k=1:5*ceil(sqrt(n))
            # gradient and drift
            cum_drift = zeros(N, 1);
            for i=1:N
                gradient = pdf.(Normal.(xtilde[i], 0.045), y) .* (y .- xtilde[i])/(0.045^2);
                cum_drift[i] = 2 * sum(hN .* gradient)/(alpha*(n+2)*(n+1)) + 2*n/(n+2) * (xtilde[i] - m0)/sigma0^2;
            end
            xtilde = xtilde .+  dt * cum_drift./(1 .+ Niter^(-a) * abs.(cum_drift)) .+ sqrt(2*dt)*randn(N, 1);
        end
        x[n+1, :] = xtilde;
    end
    time = wsample(2:(Niter+1), 2*(2:(Niter+1))./(Niter*(Niter+3)));
    return x, time
end
#= WGF for Spanish flu data
OUTPUTS
1 - particle locations
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' user selected initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'a' parameter for tamed Euler scheme
=#
function wgf_flu_tamed(N, dt, Niter, alpha, x0, m0, sigma0, muSample, M, a)
   # initialise a matrix x storing the particles
   x = zeros(Niter, N);
   # initial distribution is given as input:
   x[1, :] = x0;

   for n=1:(Niter-1)
       # get samples from h(y)
       y = sample(muSample, M, replace = true);
       # Compute h^N_{n}
       hN = zeros(M, 1);
       for j=1:M
           hN[j] = mean(0.595*pdf.(Normal(8.63, 2.56), y[j] .- x[n, :]) +
                   0.405*pdf.(Normal(15.24, 5.39), y[j] .- x[n, :]))
       end

       # gradient and drift
       drift = zeros(N, 1);
       for i=1:N
           gradient = 0.595*pdf.(Normal(8.63, 2.56), y .- x[n, i]) .* (y .- x[n, i] .- 8.63)/(2.56^2) +
                   0.405*pdf.(Normal(15.24, 5.39), y .- x[n, i]) .* (y .- x[n, i] .- 15.24)/(5.39^2);
           drift[i] = mean(gradient./hN) + alpha*(x[n, i] .- m0)/sigma0^2;
       end
       # update locations
       x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ Niter^(-a) * abs.(drift)) .+  sqrt(2*alpha*dt)*randn(N, 1);
   end
   return x
end
end
