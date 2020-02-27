
function [new_variance, new_drift] = exactAT_wgf(current_variance, X0, current_drift, lambda, sigmaG, sigmaH, dt)
new_variance = current_variance + 2*lambda/dt^2;
alphasq = 2*sigmaG^2 + 2*sigmaG*current_variance - sigmaG*sigmaH + sigmaH*current_variance;
beta = 2*sigmaG^2 + 2*sigmaG*current_variance - 2*sigmaG*sigmaH;
%     gamma = sigmaG*sigmaH + sigmaH*variance(i-1);
new_drift = current_drift + dt* (sigmaG + current_variance)/(sqrt(alphasq)*sigmaG)*...
    exp(-alphasq*beta*current_drift^2 - alphasq*(alphasq - beta)*X0^2 +...
    (current_drift*beta + X0*(alphasq - beta))^2)*...
    beta*(current_drift - X0)/alphasq;
end
