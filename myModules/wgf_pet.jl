module wgf;

using Distributions;
using Statistics;

using samplers;

export wgf_pet_tamed

#= WGF for positron emission tomography
OUTPUTS
1 - particle locations (2D)
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'lambda' regularisation parameter
'muSample' sample from noisy image μ(y)
'M' number of samples from h(y) to be drawn at each iteration
'sigma' standard deviation for Normal describing alignment
'a' parameter for tamed Euler scheme
=#
function wgf_pet_tamed(N, dt, Niter, alpha, muSample, M, sigma, a)
    # initialise two matrices x, y storing the particles
    x1 = zeros(Niter, N);
    x2 = zeros(Niter, N);
    # intial distribution
    x0 = rand(MvNormal([0, 0], 0.1*Diagonal(ones(2))), N);
    x1[1, :] = x0[1, :];
    x2[1, :] = x0[2, :];
    for n=1:(Niter-1)
        # get sample from μ(y)
        muIndex = sample(1:size(muSample, 1), M, replace = true);
        y = muSample[muIndex, :];
        # Compute h^N_{n}
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(0, sigma), x1[n, :] * cos(y[j, 1]) .+
                    x2[n, :] * sin(y[j, 1]) .- y[j, 2])
                    );
        end
        # gradient and drift
        driftX1 = zeros(N, 1);
        driftX2 = zeros(N, 1);
        for i=1:N
            # precompute common quantities for gradient
            prec = -pdf.(Normal.(0, sigma), x1[n, i] * cos.(y[:, 1]) .+
                    x2[n, i] * sin.(y[:, 1]) .- y[:, 2]) .*
                    (x1[n, i] * cos.(y[:, 1]) .+
                    x2[n, i] * sin.(y[:, 1]) .- y[:, 2])/sigma^2;
            gradientX1 = prec .* cos.(y[:, 1]);
            gradientX2 = prec .* sin.(y[:, 1]);
            # keep only finite elements
            g1h = gradientX1./hN;
            g2h = gradientX2./hN;
            g1h[(!).(isfinite.(g1h))] .= 0;
            g2h[(!).(isfinite.(g2h))] .= 0;
            driftX1[i] = mean(g1h);
            driftX2[i] = mean(g2h);
        end
        # update locations
        drift_norm = sqrt.(sum([driftX1 driftX2].^2, dims = 2));
        x1[n+1, :] = x1[n, :] .+ dt * driftX1./(1 .+ Niter^(-a) * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
        x2[n+1, :] = x2[n, :] .+ dt * driftX2./(1 .+ Niter^(-a) * drift_norm) .+ sqrt(2*alpha*dt)*randn(N, 1);
    end
    return x1, x2
end
end
