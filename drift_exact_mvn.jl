function drift_exact_mvn(mu, sigma0, sigmaG, sigmaH, x)
    N = size(x, 2);

    sigma = sigma0 + sigmaG;
    rhoH = sigmaH[1, 2]/sqrt(sigmaH[1, 1] * sigmaH[2, 2]);
    rhoG = sigmaG[1, 2]/sqrt(sigmaG[1, 1] * sigmaG[2, 2]);
    rho = sigma[1, 2]/sqrt(sigma[1, 1] * sigma[2, 2]);

    alpha2 = 1/((1-rhoH^2) * sigmaH[1, 1]) + 1/((1-rhoG^2) * sigmaG[1, 1]) -
        1/((1-rho^2) * sigma[1, 1]);

    A = (rhoH * (1-rho^2) * sqrt(sigma[1, 1] * sigma[2, 2]) -
        rho * (1-rhoH^2) * sqrt(sigmaH[1, 1] * sigmaH[2, 2]))/
        ((1-rho^2) * sqrt(sigma[1, 1] * sigma[2, 2]) *
        (1-rhoH^2) * sqrt(sigmaH[1, 1] * sigmaH[2, 2]));
    B = rhoG/((1-rhoG^2) * sqrt(sigmaG[1, 1] * sigmaG[2, 2]));
    C = ((1-rho^2) * sigma[1, 1] - (1-rhoH^2) * sigmaH[1, 1])/
        ((1-rho^2) * sigma[1, 1] * (1-rhoH^2) * sigmaH[1, 1]);
    D = 1/((1-rhoG^2) * sigmaG[1, 1]);

    beta2 =  1/((1-rhoH^2) * sigmaH[2, 2]) + 1/((1-rhoG^2) * sigmaG[2, 2]) -
        1/((1-rho^2) * sigma[2, 2]) - (A^2 + B^2 + 2*A*B)/alpha2;

    E = 1/((1-rhoH^2) * sigmaH[2, 2]) - 1/((1-rho^2) * sigma[2, 2]) -
        (A^2 + 2*A*B)/alpha2;
    F = 1/((1-rhoG^2) * sigmaG[2, 2]) - (B^2 + 2*A*B)/alpha2;
    G = (A + B) * C/alpha2 - 1/((1-rhoH^2) * sqrt(sigmaH[1, 1] * sigmaH[2, 2])) +
        1/((1-rho^2) * sqrt(sigma[1, 1] * sigma[2, 2]));
    H = (A + B) * D/alpha2 - 1/((1-rhoG^2) * sqrt(sigmaG[1, 1] * sigmaG[2, 2]));

    common_term = zeros(N, N);
    for i=1:N
        one = exp((C^2 * mu[1]^2 + D^2 * x[i, 1]^2)/(2*alpha2) -
            0.5 * mu[1]^2 * (1/((1-rhoH^2) * sigmaH[1, 1]) - 1/(1-rho^2) * sigma[1, 1]) -
            0.5 * x[i, 1]^2 /((1-rhoG^2) * sigmaG[1, 1]) +
            mu[1] * mu[2] * (rhoH/((1-rhoH^2) * sqrt(sigmaH[1, 1] * sigmaH[2, 2])) -
            rho/((1-rho^2) * sqrt(sigma[1, 1] * sigma[2, 2])))
            );
        for j=1:N
            common_term[N-i+1, j] = one * rhoG * x[i, 1] * x[j, 2]/((1-rhoG^2) *
            sqrt(sigmaG[1, 1] * sigmaG[2, 2]));

        end
    end
    common_term = common_term * sqrt(2*pi * sigma[1, 1] * sigma[2, 2] * (1-rho^2))/
        (sqrt((1-rhoH^2) * sigmaH[1, 1] * sigmaH[2, 2] *
        (1-rhoG^2) * sigmaG[1, 1] * sigmaG[2, 2] * alpha2) * (1-rhoG^2)) *
        sqrt(2*pi/beta2);

    two =  exp.((A^2 * mu[2]^2 .+ B^2 * x[:, 2].^2)/(2*beta2) .-
    0.5 * mu[2]^2 * (1/((1-rhoH^2) * sigmaH[2, 2]) - 1/(1-rho^2) * sigma[2, 2]) .-
    0.5 * x[:, 2].^2 /((1-rhoG^2) * sigmaG[2, 2]) .+
    2/alpha2 * (A*B*mu[2]*x[:, 2] .- A*C*mu[1]*mu[2] .- A*D*mu[2]*x[:, 1]
    .- C*B*mu[1]*x[:, 2] .- D*B*x[:, 1].*x[:, 2])
    );

    common_term = one .* two .* exp.((E*mu[2] .+ F*x[:, 2] .+ G*mu[1] .+ H*x[:, 1]).^2/
    (2*beta2)) .* (x[1]/sqrt(sigmaG[1, 1]) .- x[:, 2]/sqrt(sigmaG[2, 2]) .+
    (C*mu[1] .+ D*x[:, 1] .+ A*mu[2] .+ B*x[:, 2])/(alpha2 * sqrt(sigmaG[1, 1])) +
    (1/sqrt(sigmaG[2, 2]) - (A + B)/(alpha2 * sqrt(sigmaG[1, 1]))) *
    (E*mu[2] .+ F*x[:, 2] .+ G*mu[1] .+ H*x[:, 1])/beta2);

    drift1 = -common_term/sqrt(sigmaG[1, 1]);
    drift2 = common_term/sqrt(sigmaG[2, 2]);
    return drift1, drift2
end
