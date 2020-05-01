module wgfserver;

using Distributions;
using Statistics;

export wgf_AT
export wgf_AT_15
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
function wgf_AT_15(N, Niter, lambda, x0, M)
    # time step
    dt = 1/Niter;
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # get samples from h(y)
    y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);

    for n=1:(Niter-1)
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end
        # gradient and drift
        drift1 = zeros(N, 1);
        drift2 = zeros(N, 1);
        drift3 = zeros(N, 1);
        for i=1:N
            # common quantity
            prec = pdf.(Normal.(x[n, i], 0.045), y)/(0.045^2);
            drift1[i] = mean((prec .* (y .- x[n, i]) )./hN);
            drift2[i] = mean((prec .* ((y .- x[n, i]).^2/(0.045^2) .- 1))./hN);
            drift3[i] = mean((prec .* (y .- x[n, i])/(0.045^2) .*
                ((y .- x[n, i]).^2/(0.045^2) .- 3))./hN);
        end
        # update locations
        # BM
        u1 = randn(N, 1);
        u2 = randn(N, 1);
        W = sqrt(dt)*u1;
        Z = 0.5*dt^(3/2) * (u1 .+ u2/sqrt(3));
        x[n+1, :] = x[n, :] .+ drift1*dt .+ sqrt(2*lambda)*W .+
            sqrt(2*lambda)*drift2 .* Z .+
            0.5*dt^2*(drift1 .* drift2 .+ lambda*drift3);
    end
    return x
end

end
