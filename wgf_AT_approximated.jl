function wgf_AT_approximated(dt, lambda, M, N)

    Niter = trunc(Int, 1/dt);
    x = zeros(Niter, N);
    drift = zeros(Niter-1, N);
    y = rand(Normal(0.5, sqrt(0.043^2 + 0.045^2)), M);
    x[1, :] = rand(1, N);
    for n=1:(Niter-1)
        hN = zeros(M, 1);
        for j=1:M
            hN[j] = mean(pdf.(Normal.(x[n, :], 0.045), y[j]));
        end

        for i=1:N
            gradient = pdf.(Normal.(x[n, i], 0.045), y) .* (y .- x[n, i])/(0.045^2);
            drift[n, i] = mean(gradient./hN);
        end
        x[n+1, :] = x[n, :] .+ drift[n, :]*dt .+ sqrt(2*lambda)*dt*randn(N, 1);
    end
    return x, drift
end

function exact_minimizer(sigmaG, sigmaH, lambda)
    variance  = (sigmaH - sigmaG .+ 2*lambda*sigmaG +
                sqrt.(sigmaG^2 + sigmaH^2 .- 2*sigmaG*sigmaH.*(1 .- 2*lambda)))./
                (2*(1 .- lambda));
    return variance
end
