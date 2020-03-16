function drift_exact(μ, sigma0, sigmaG, sigmaH, x)
    drift = (sigmaG + sigma0 - sigmaH)*(sigmaG + sigma0)/
            (sigmaG^2 + sigmaG*sigma0 + sigma0*sigmaH)^(3/2) *
            exp.(-(sigmaG + sigma0 - sigmaH)*(μ .- x).^2 ./
            (2*(sigmaG^2 + sigmaG*sigma0 + sigma0*sigmaH))) .* (μ .- x);
end
