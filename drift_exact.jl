function drift_exact(μ, sigma0, sigmaG, sigmaH, x)
    β = 2*sigmaG^2 + 2*sigmaG*sigma0 - 2*sigmaG*sigmaH;
    γ = sigmaH*(sigmaG + sigma0);
    drift = (2*sigmaG + 2*sigma0 - 2*sigmaH)/(sqrt(β + γ)*sigmaG*sigmaH) *
        exp.(-(μ .- x).^2 * β*γ ./(2*sigmaG*γ*(β + γ))) .* (μ .- x);
    return drift;
end
