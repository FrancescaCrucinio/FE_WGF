function drift_exact(μ, sigma0, sigmaG, sigmaH, x)
    α = 2*sigmaG^2 + 2*sigmaG*sigma0 - sigmaG*sigmaH + sigmaH*sigma0;
    β = 2*sigmaG^2 + 2*sigmaG*sigma0 - 2*sigmaG*sigmaH;
    drift = (sigmaG + sigma0)/(sigmaG*sqrt(α)) *
    exp.((-μ^2*β .- (α-β).*x.^2 + (μ*β .+ x.*(α - β)).^2)./(2*sigmaG*sigmaH*(sigmaG+sigma0))) .*
    (μ*β .- x*β)./α;
    return drift;
end
