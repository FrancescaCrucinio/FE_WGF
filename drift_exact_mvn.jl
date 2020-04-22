function drift_exact_mvn(mu, sigma0, sigmaG, sigmaH, x, y)
    N = length(x);

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
        (A^2 +A*B)/alpha2;
    F = 1/((1-rhoG^2) * sigmaG[2, 2]) - (B^2 + A*B)/alpha2;
    G = (A + B) * C/alpha2 - rhoH/((1-rhoH^2) * sqrt(sigmaH[1, 1] * sigmaH[2, 2])) +
        rho/((1-rho^2) * sqrt(sigma[1, 1] * sigma[2, 2]));
    H = (A + B) * D/alpha2 - rhoG/((1-rhoG^2) * sqrt(sigmaG[1, 1] * sigmaG[2, 2]));

    common_term = zeros(N, N);
    for i=1:N
        first = exp(0.5*x[i]^2 * (D^2/(alpha2) - 1/((1-rhoG^2) * sigmaG[1, 1])) -
            x[i]*A*D*mu[2]/alpha2);
        for j=1:N
            meanpart = x[i]/sqrt(sigmaG[1, 1]) - y[j]/sqrt(sigmaG[2, 2]) -
                (C*mu[1]+D*x[i]-A*mu[2]-B*y[j])/(alpha2 * sqrt(sigmaG[1, 1])) +
                (1/sqrt(sigmaG[2, 2]) - (A+B)/(alpha2* sqrt(sigmaG[1, 1]))) *
                (E*mu[2]+F*y[j]+G*mu[1]+H*x[i])/beta2;
            common_term[N-i+1, j] = meanpart * first * exp(x[i]*y[j] *
            (rhoG/((1-rhoG^2) * sqrt(sigmaG[1, 1] * sigmaG[2, 2])) - B*D/alpha2) +
            0.5*y[j]^2 * (-1/((1-rhoG^2) * sigmaG[2, 2]) + B^2/alpha2) +
            y[j] * (A*B*mu[2] - B*C*mu[1])/alpha2) *
            exp((E*mu[2]+F*y[i]+G*mu[1]+H*x[i])^2/(2*beta2));
        end
    end
    constant = sqrt(sigma[1, 1] * sigma[2, 2] * (1-rho^2))/
        (sqrt((1-rhoH^2) * sigmaH[1, 1] * sigmaH[2, 2] *
        sigmaG[1, 1] * sigmaG[2, 2])  * (1-rhoG^2)^(3/2) *
        sqrt(alpha2 * beta2));
    constant_exponential = exp(0.5*mu[1]^2 * (C^2 /(alpha2) -
        1/((1-rhoH^2) * sigmaH[1, 1]) + 1/((1-rho^2) * sigma[1, 1])) +
        0.5*mu[2]^2 * (A^2 /(2*alpha2) - 1/((1-rhoH^2) * sigmaH[2, 2])
        + 1/((1-rho^2) * sigma[2, 2])) +
        mu[1]*mu[2] * (rhoH/((1-rhoH^2) * sqrt(sigmaH[1, 1] * sigmaH[2, 2])) -
        rho/((1-rho^2) * sqrt(sigma[1, 1] * sigma[2, 2])) - A*C/alpha2));

    common_term = common_term * constant_exponential * constant;

    drift1 = -common_term/sqrt(sigmaG[1, 1]);
    drift2 = common_term/sqrt(sigmaG[2, 2]);
    return drift1, drift2
end
