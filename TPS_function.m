function Omega_hat = TPS_function(R, varargin)
% HD_ITPS_free_cov
%
% Parameter-free high-dimensional calibrated ITPS / PCTPS.
%
% METHODS
%   method = 'ITPS'
%       Target precision matrix:
%           H = I
%
%       alpha:
%           v_I  = ||C - I||_F^2 / N
%           c_0  = (N - 1) / (T - 1)
%           alpha = c_0 / (v_I - c_0), if v_I > c_0
%           alpha = Inf, otherwise
%
%   method = 'PCTPS'
%       Positive-correlation target:
%           T_rho = (1-rho) I + rho 11'
%           H_rho = inv(T_rho)
%
%       rho:
%           rho = max( average off-diagonal correlation, 0 )
%
%       alpha:
%           C_tilde = T_rho^{-1/2} C T_rho^{-1/2}
%           v_rho   = ||C_tilde - I||_F^2 / N
%           c_rho   = (N + 1) / (T - 1)
%           alpha   = c_rho / (v_rho - c_rho), if v_rho > c_rho
%           alpha   = Inf, otherwise
%
% INPUT
%   R : T-by-N return matrix.
%
% OUTPUT
%   Omega_hat : N-by-N precision matrix estimator.

% -------------------------------------------------------------------------
% Optional parameters
% -------------------------------------------------------------------------
p = inputParser;

addParameter(p, 'method', 'ITPS');
addParameter(p, 'demean', true);
addParameter(p, 'missingMethod', 'dropRows');
% 'dropRows'  : remove rows containing NaN or Inf
% 'assetMean' : replace missing values by asset-wise sample mean

addParameter(p, 'forceUnitDiag', true);
% If true, the estimated standardised covariance matrix is rescaled to
% have unit diagonal before returning to the original covariance scale.

addParameter(p, 'rhoMax', 1 - 1e-8);
% Numerical upper bound for rho in PCTPS.

addParameter(p, 'minStd', 1e-12);
addParameter(p, 'minEig', 1e-10);

parse(p, varargin{:});

method        = upper(char(p.Results.method));
demean        = p.Results.demean;
missingMethod = lower(p.Results.missingMethod);
forceUnitDiag = p.Results.forceUnitDiag;
rhoMax        = p.Results.rhoMax;
minStd        = p.Results.minStd;
minEig        = p.Results.minEig;

% -------------------------------------------------------------------------
% Data cleaning
% -------------------------------------------------------------------------
R = double(R);

switch missingMethod
    case 'droprows'
        R = R(all(isfinite(R), 2), :);

    case 'assetmean'
        R = fill_missing_by_asset_mean(R);

    otherwise
        error('missingMethod must be either ''dropRows'' or ''assetMean''.');
end

[T, N] = size(R);

if T < 3
    error('The return matrix must contain at least three valid observations.');
end

if N < 2
    error('The return matrix must contain at least two assets.');
end

% -------------------------------------------------------------------------
% Demean and standardise returns
% -------------------------------------------------------------------------
if demean
    mu = mean(R, 1);
else
    mu = zeros(1, N);
end

Rc = R - mu;

sigma = sqrt(sum(Rc.^2, 1) / (T - 1));
sigma(sigma < minStd) = minStd;

Z = bsxfun(@rdivide, Rc, sigma);

% -------------------------------------------------------------------------
% Sample correlation matrix
% -------------------------------------------------------------------------
C = (Z' * Z) / (T - 1);
C = (C + C') / 2;
C(1:N+1:end) = 1;

I = eye(N);
oneN = ones(N, 1);

% -------------------------------------------------------------------------
% Construct target and parameter-free alpha
% -------------------------------------------------------------------------
switch method

    case 'ITPS'

        T_target = I;
        H_target = I;

        v = norm(C - I, 'fro')^2 / N;
        c_noise = (N - 1) / (T - 1);


    case {'PCTPS', 'PC-TPS', 'PC_TPS'}

        % Average off-diagonal correlation
        rhoHat = (sum(C(:)) - trace(C)) / (N * (N - 1));
        rho = max(rhoHat, 0);
        rho = min(rho, rhoMax);

        % T_rho = (1-rho)I + rho 11'
        T_target = (1 - rho) * I + rho * (oneN * oneN');

        % Closed-form inverse of equicorrelation matrix
        aInv = 1 / (1 - rho);
        bInv = rho / ((1 - rho) * (1 + (N - 1) * rho));

        H_target = aInv * I - bInv * (oneN * oneN');
        H_target = (H_target + H_target') / 2;

        % Target whitening:
        % T^{-1/2} = p_perp * P_perp + p_m * uu'
        u = oneN / sqrt(N);
        Pm = u * u';
        Pp = I - Pm;

        eigMarket = 1 + (N - 1) * rho;
        eigPerp   = 1 - rho;

        T_inv_sqrt = (1 / sqrt(eigPerp)) * Pp ...
                   + (1 / sqrt(eigMarket)) * Pm;

        C_tilde = T_inv_sqrt * C * T_inv_sqrt;
        C_tilde = (C_tilde + C_tilde') / 2;

        v = norm(C_tilde - I, 'fro')^2 / N;

        % For target-whitened covariance, diagonal terms also fluctuate.
        c_noise = (N + 1) / (T - 1);

    otherwise
        error('method must be either ''ITPS'' or ''PCTPS''.');
end

% -------------------------------------------------------------------------
% Parameter-free high-dimensional alpha
%
% alpha = c_noise / (v - c_noise), if v > c_noise
% alpha = Inf, otherwise
% -------------------------------------------------------------------------
if v > c_noise
    alpha = c_noise / (v - c_noise);
    isTargetOnly = false;
else
    alpha = Inf;
    isTargetOnly = true;
end

% -------------------------------------------------------------------------
% Closed-form ITPS / PCTPS solution
%
% General form:
%   min tr(C Omega) - logdet(Omega) + alpha/2 ||Omega - H||_F^2
%
% FOC:
%   C - Omega^{-1} + alpha(Omega - H) = 0
%
% Let:
%   A = C - alpha H
%   A = V diag(a) V'
%
% Then:
%   theta_k = (-a_k + sqrt(a_k^2 + 4 alpha)) / (2 alpha)
%
% If alpha = Inf, the estimator collapses to the target:
%   Omega = H
%   C_hat = T
% -------------------------------------------------------------------------
if isTargetOnly

    C_hat = T_target;


else

    A = C - alpha * H_target;
    A = (A + A') / 2;

    [V, D_A] = eig(A);
    aVals = real(diag(D_A));

    theta = (-aVals + sqrt(aVals.^2 + 4 * alpha)) ./ (2 * alpha);
    theta = max(theta, minEig);

    dVals = 1 ./ theta;

    C_hat = V * diag(dVals) * V';
    C_hat = (C_hat + C_hat') / 2;
end

% -------------------------------------------------------------------------
% Numerical PSD repair
% -------------------------------------------------------------------------
C_hat = psd_floor(C_hat, minEig);

% Force to correlation matrix if requested
if forceUnitDiag
    C_hat = force_correlation_matrix(C_hat, minEig);
end

Omega_C_final = inv_spd(C_hat, minEig);

% -------------------------------------------------------------------------
% Original-scale precision matrix
% -------------------------------------------------------------------------
Omega_hat = Omega_C_final ./ (sigma(:) * sigma(:)');
Omega_hat = (Omega_hat + Omega_hat') / 2;




end


% =========================================================================
% Helper function: fill missing values by asset-wise sample mean
% =========================================================================
function R = fill_missing_by_asset_mean(R)

[~, N] = size(R);

for j = 1:N
    x = R(:, j);
    ok = isfinite(x);

    if ~any(ok)
        error('Asset %d contains no finite observations.', j);
    end

    colMean = mean(x(ok));
    x(~ok) = colMean;
    R(:, j) = x;
end

if any(~isfinite(R(:)))
    error('The return matrix still contains invalid values after filling.');
end

end


% =========================================================================
% Helper function: inverse of symmetric positive definite matrix by eig
% =========================================================================
function A_inv = inv_spd(A, floorEig)

A = (A + A') / 2;

[V, D] = eig(A);
d = real(diag(D));
d = max(d, floorEig);

A_inv = V * diag(1 ./ d) * V';
A_inv = (A_inv + A_inv') / 2;

end


% =========================================================================
% Helper function: PSD eigenvalue flooring
% =========================================================================
function A_psd = psd_floor(A, floorEig)

A = (A + A') / 2;

[V, D] = eig(A);
d = real(diag(D));
d = max(d, floorEig);

A_psd = V * diag(d) * V';
A_psd = (A_psd + A_psd') / 2;

end


% =========================================================================
% Helper function: force a PSD matrix into a correlation-like matrix
% =========================================================================
function C_corr = force_correlation_matrix(C, floorEig)

C_corr = (C + C') / 2;
C_corr = psd_floor(C_corr, floorEig);

d = diag(C_corr);
d = max(d, floorEig);

C_corr = bsxfun(@rdivide, C_corr, sqrt(d));
C_corr = bsxfun(@rdivide, C_corr, sqrt(d)');
C_corr = (C_corr + C_corr') / 2;

C_corr = psd_floor(C_corr, floorEig);

d = diag(C_corr);
d = max(d, floorEig);

C_corr = bsxfun(@rdivide, C_corr, sqrt(d));
C_corr = bsxfun(@rdivide, C_corr, sqrt(d)');
C_corr = (C_corr + C_corr') / 2;

end
