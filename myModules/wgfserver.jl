module wgfserver;

using Distributions;
using Statistics;

export wgf_AT
export wgf_gaussian_mixture

#=
 WGF for analytically tractable example (approximated drift)
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'Niter' number of time steps
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
=#
function wgf_AT(N, Niter, lambda, x0, M)
    # time step
    dt = 1/Niter;
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # initialise a matrix drift storing the drift
    drift = zeros(Niter-1, N);
    # get samples from h(y)
    y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);

    for n=1:(Niter-1)
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # gradient and drift
        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
            drift[n, i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
    end
    return x, drift
end

#=
 WGF for gaussian mixture
OUTPUTS
1 - particle locations
2 - drift evolution
INPUTS
'N' number of particles
'Niter' number of time steps
'lambda' regularisation parameter
'x0' user selected initial distribution
'M' number of samples from h(y) to be drawn at each iteration
=#
function wgf_gaussian_mixture(N, Niter, lambda, x0, M)
    # time step
    dt = 1/Niter;
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # initialise a matrix drift storing the drift
    drift = zeros(Niter-1, N);
    # get samples from h(y)
    y = Ysample_gaussian_mixture(M);

    for n=1:(Niter-1)
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # gradient and drift
        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
            drift[n, i] = mean(gradient./hN);
        end
        # update locations
        x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda*dt)*randn(N, 1);
    end
    return x, drift
end

end
