function drift_exact_mvn_mean0(sigma0, sigmaG, sigmaH, x, y)
    N = length(x);

    sigma = sigma0 + sigmaG;
    rhoH = sigmaH[1, 2]/sqrt(sigmaH[1, 1] * sigmaH[2, 2]);
    rhoG = sigmaG[1, 2]/sqrt(sigmaG[1, 1] * sigmaG[2, 2]);
    rho = sigma[1, 2]/sqrt(sigma[1, 1] * sigma[2, 2]);

    alpha2 = 1/((1-rhoH^2) * sigmaH[1, 1]) + 1/((1-rhoG^2) * sigmaG[1, 1]) -
        1/((1-rho^2) * sigma[1, 1]);

    B = 1/((1-rhoG^2) * sigmaG[1, 1]);
    C = -rhoG/((1-rhoG^2) * sqrt(sigmaG[1, 1] * sigmaG[2, 2]));
    A = rhoH/((1-rhoH^2) * sqrt(sigmaH[1, 1] * sigmaH[2, 2])) -
        rho/((1-rho^2) * sqrt(sigma[1, 1] * sigma[2, 2])) - C;


    beta2 =  1/((1-rhoH^2) * sigmaH[2, 2]) + 1/((1-rhoG^2) * sigmaG[2, 2]) -
        1/((1-rho^2) * sigma[2, 2]) + A^2/alpha2;

    D = B*A/alpha2 + C;
    E = A*C/alpha2 + 1/((1-rhoG^2)sigmaG[2, 2]);

    common_term = zeros(N, N);
    for i=1:N
        first = exp(0.5*x[i]^2*(-B+B/alpha2));
        for j=1:N
            meanpart = x[i] * (1/sqrt(sigmaG[1, 1]) -
                B/(alpha2 * sqrt(sigmaG[1, 1])) +
                D/(beta2 * sqrt(sigmaG[2, 2])) -
                A*D/(alpha2 * beta2 * sqrt(sigmaG[1, 1]))) +
                y[j] * (-1/sqrt(sigmaG[2, 2]) -
                C/(alpha2 * sqrt(sigmaG[1, 1])) +
                E/(beta2 * sqrt(sigmaG[2, 2])) -
                A*E/(alpha2 * beta2 * sqrt(sigmaG[1, 1])));
            common_term[N-i+1, j] = meanpart * first*
                exp(x[i]*y[j] * (-C+B*C/alpha2) +
                0.5*y[j]^2 * (-1/((1-rhoG^2) * sigmaG[2, 2]) + C^2/alpha2));
        end
    end
    constant = sqrt(sigma[1, 1] * sigma[2, 2] * (1-rho^2))/
        (sqrt((1-rhoH^2) * sigmaH[1, 1] * sigmaH[2, 2] *
        sigmaG[1, 1] * sigmaG[2, 2])  * (1-rhoG^2)^(3/2) *
        sqrt(alpha2 * beta2));

    common_term = common_term * constant;

    drift1 = -common_term/sqrt(sigmaG[1, 1]);
    drift2 = common_term/sqrt(sigmaG[2, 2]);
    return drift1, drift2
end
