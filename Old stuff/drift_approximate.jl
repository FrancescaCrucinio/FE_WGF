function drift_approximate(μ, sigma0, sigmaG, sigmaH, x, M, N)
    l = length(x);
    y = rand(Normal(μ, sqrt(sigmaH)), M);
    hN = zeros(M, 1);
    x0 = rand(Normal(μ, sqrt(sigma0)), N);
    for j=1:M
        hN[j] = mean(pdf.(Normal.(x0, sqrt(sigmaG)), y[j]));
    end

    drift = zeros(l, 1);
    for i=1:l
        gradient = pdf.(Normal.(x[i], sqrt(sigmaG)), y) .* (y .- x[i])/sigmaG;
        drift[i] = mean(gradient./hN);
    end
    return drift;
end
