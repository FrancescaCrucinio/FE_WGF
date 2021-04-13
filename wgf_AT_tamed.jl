#= WGF for analytically tractable example
OUTPUTS
1 - particle locations
2 - value of functional E at each iteration
INPUTS
'N' number of particles
'dt' discretisation step
'Niter' number of iterations
'alpha' regularisation parameter
'x0' initial distribution
'm0' mean of prior
'sigma0' standard deviation of prior
'muSample' sample from μ
'M' number of samples from μ(y) to be drawn at each iteration
'reference' reference measure π0
=#
function wgf_AT_tamed(N, dt, Niter, alpha, x0, m0, sigma0, muSample, M, reference)
    # initialise a matrix x storing the particles
    x = zeros(Niter, N);
    # initial distribution is given as input:
    x[1, :] = x0;
    # value of functional
    E = zeros(Niter, 1);
    # parameters for priors
    nu = 100;
    b = sigma0/sqrt(2);
    for n=1:(Niter-1)
        # log-likelihood
        loglik = zeros(1, length(muSample));
        for i=1:length(muSample)
            loglik[i] = mean(pdf.(Normal.(x[n, :], 0.45), muSample[i]));
        end
        loglik = -log.(loglik);
        kl = mean(loglik);
        # entropy
        Rpihat = rks.kde(x = x[n, :], var"eval.points" = x[n, :] );
        pihat = abs.(rcopy(Rpihat[3]));
        # prior
        if(reference == "uniform")
            kl_prior = mean(log.(pihat));
        elseif(reference == "normal")
            prior = pdf.(Normal(0, sigma0), x[n, :]);
            kl_prior = mean(log.(pihat./prior));
        elseif(reference == "t")
            prior = pdf.(TDist(nu), x[n, :]);
            kl_prior = mean(log.(pihat./prior));
        elseif(reference == "Laplace")
            prior = pdf.(Laplace(0, b), x[n, :]);
            kl_prior = mean(log.(pihat./prior));
        end

        E[n] = kl+alpha*kl_prior;
        # get samples from μ(y)
        y = sample(muSample, M, replace = false);
        # Compute μ^N_{n}
        muN = zeros(M, 1);
        for j=1:M
            muN[j] = mean(pdf.(Normal.(x[n, :], 0.45), y[j]));
        end
        # gradient and drift
        drift = zeros(N, 1);
        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.45), y) .* (y .- x[n, i])/(0.45^2);
            drift[i] = mean(gradient./muN);
        end
        if(reference == "normal")
            drift = drift .+ alpha*x[n, :]/sigma0^2;
        elseif(reference == "t")
            drift = drift .+ alpha*(-nu + 1)*x[n, :]./(nu .+ x[n, :].^2);
        elseif(reference == "Laplace")
            drift = drift .- alpha*sign.(x[n, :])/b;
        end

        # update locations
        x[n+1, :] = x[n, :] .+ dt * drift./(1 .+ dt * abs.(drift)) .+  sqrt(2*alpha*dt)*randn(N, 1);
    end

    loglik = zeros(1, length(muSample));
    for i=1:length(muSample)
        loglik[i] = mean(pdf.(Normal.(x[Niter, :], 0.45), muSample[i]));
    end
    loglik = -log.(loglik);
    kl = mean(loglik);
    # entropy
    Rpihat = rks.kde(x = x[Niter, :], var"eval.points" = x[Niter, :] );
    pihat = abs.(rcopy(Rpihat[3]));
    # prior
    if(reference == "uniform")
        kl_prior = mean(log.(pihat));
    elseif(reference == "normal")
        prior = pdf.(Normal(0, sigma0), x[Niter, :]);
        kl_prior = mean(log.(pihat./prior));
    elseif(reference == "t")
        prior = pdf.(TDist(nu), x[Niter, :]);
        kl_prior = mean(log.(pihat./prior));
    elseif(reference == "Laplace")
        prior = pdf.(Laplace(0, b), x[Niter, :]);
        kl_prior = mean(log.(pihat./prior));
    end

    E[Niter] = kl+alpha*kl_prior;
    return x, E
end
