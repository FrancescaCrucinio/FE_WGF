
function pda_gaussian_mixture_tamed(N, dt, Niter, alpha, x0, sigma0, m0, muSample, M)
    # initialise a matrix x storing the particles
    x = zeros(Niter+1, N);
    # initial distribution
    x[1, :] = x0;

    muN = zeros(M, 1);
    for n=1:Niter
        # samples from μ(y)
        y = sample(muSample, M, replace = false);
        # Compute μ^N_{n}
        for j=1:M
            muN[j] = muN[j] + n/mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end

        # Langevin steps
        xtilde = x[n, :];
        for k=1:10
            # gradient and drift
            drift = zeros(N, 1);
            for i=1:N
                gradient = pdf.(Normal.(xtilde[i], 0.045), y) .* (y .- xtilde[i])/(0.045^2);
                drift[i] = 2 * mean(muN .* gradient)/(alpha*(n+2)*(n+1)) + 2*n/(n+2) * (xtilde[i] - m0)/sigma0^2;
            end
            xtilde = xtilde .+  dt * drift./(1 .+ dt * abs.(drift)) .+ sqrt(2*dt)*randn(N, 1);
        end
        x[n+1, :] = xtilde;
    end
    time = wsample(2:(Niter+1), 2*(2:(Niter+1))./(Niter*(Niter+3)));
    return x, time
end


x_pda, time =  pda_gaussian_mixture_tamed(Nparticles, dt, Niter, alpha, x0, sigma0, m0, muSample, M);

RKDEyWGF = rks.kde(x = x_pda[time, :], var"eval.points" = KDEx);
KDEyWGF_pda = abs.(rcopy(RKDEyWGF[3]));

plot(KDEx, pi.(KDEx))
plot!(KDEx, KDEyWGF_pda)
