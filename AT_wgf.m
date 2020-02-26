function [x]= AT_wgf(N, M, Niter, lambda)

x = zeros(Niter, N);

x(1, :) = 0.1*ones(1, N);
for n=1:Niter
    % get samples from h(y)
    y = 0.5 + sqrt(0.043^2 + 0.045^2) * randn(M, 1);
    % Compute h^N_{n}
    hN = zeros(M,1);
    for j=1:M
        hN(j) = sum(normpdf(y(j), x(n, :), 0.045));
    end
    drift = zeros(1, N);
    for i=1:N
        gradient = normpdf(y, x(n, i), 0.045) .* (y - x(n, i))/(0.045^2);
        drift(i) = sum(gradient./hN);
    end
    x(n+1, :) = x(n, :) - drift/Niter + sqrt(2*lambda/Niter)*randn(1, N);
end
end