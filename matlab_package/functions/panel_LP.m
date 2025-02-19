function [LP_estimate, LP_CI] = panel_LP(y, sX, W, FEs, i_index, t_index, H, n_lag_max, small_sample, cumulative)
% PANEL_LP: Panel local projection estimates and confidence intervals.
% 
% This function estimates impulse responses using local projections (LP)
% in a panel data setting and provides confidence intervals recomended in
% "Micro Responses to Macro Shocks" (Almuzara, Sancibrian). 
% It supports small-sample refinements, lagged controls, and cumulative responses.
%
% USAGE:
%   [LP_estimate, LP_CI] = panel_LP(y, sX, W, dummies, i_index, t_index, H, n_lag_max, small_sample, cumulative)
%
% INPUTS:
%   y            : (T*N x 1) Dependent variable vector.
%   sX           : (T*N x S) Shock variable(s) matrix.
%   W            : (T*N x W) Control variables matrix.
%   FEs          : (T*N x D) Fixed effects/dummy variables matrix.
%   i_index      : (T*N x 1) Individual identifiers.
%   t_index      : (T*N x 1) Time period identifiers.
%   H            : (scalar, optional) Maximum horizon. Default: ceil(0.25*T).
%   n_lag_max    : (scalar, optional) Maximum number of lags. Default: ceil((T-H)^(1/3)).
%   small_sample : (logical, optional) Apply small-sample refinement. Default: true.
%   cumulative   : (logical, optional) Compute cumulative responses. Default: false.
%
% OUTPUTS:
%   LP_estimate  : ((H+1) x S) Matrix of LP estimates for each horizon.
%   LP_CI        : Structure containing:
%                  - SE   : ((H+1) x S) Standard errors.
%                  - df   : ((H+1) x 1) Degrees of freedom for inference.
%                  - CI90 : ((H+1) x S x 2) 90% confidence intervals.
%                  - CI95 : ((H+1) x S x 2) 95% confidence intervals.
%                  - CI99 : ((H+1) x S x 2) 99% confidence intervals.
%
% DEPENDENCIES:
%   - time_shift.m: Used to shift time series.
%   - regress_HDFE.m: Performs high-dimensional fixed effects regression.
%
% NOTES:
%   - The function estimates impulse responses up to horizon H.
%   - Small-sample refinements follow Imbens & Kolesar (2016).
%   - Time-clustered standard errors are computed if small_sample = false.
%
% EXAMPLE USAGE:
%   [LP_est, LP_CI] = panel_LP(y, sX, W, dummies, i_index, t_index, 10, 3, true, false);
%
% Version: 2024 Jun 10

% Recover default options
if (nargin < 7)
    H = ceil(0.25*length(unique(t_index)));
end
if (nargin < 8)
    n_lag_max = ceil((length(unique(t_index))-H)^(1/3));
end
if (nargin < 9) 
    small_sample = true;  % apply Imbens-Kolesar small-sample refinement
end
if (nargin < 10) 
    cumulative = false; % compute cumulative impulse responses
end

% Preallocate output
n_s         = size(sX, 2);
LP_estimate = zeros(H+1, n_s);
LP_CI       = struct();
LP_CI.SE    = zeros(H+1, n_s);
LP_CI.df    = zeros(H+1, 1);
LP_CI.CI90  = zeros(H+1, n_s, 2);
LP_CI.CI95  = zeros(H+1, n_s, 2);
LP_CI.CI99  = zeros(H+1, n_s, 2);

% Iterate over horizons
y_h = zeros(length(i_index), 1);
for h = 0:H

    % Lead regressand
    if (cumulative == true)
        y_h = y_h + time_shift(y, i_index, t_index, h);
    else
        y_h = time_shift(y, i_index, t_index, h);
    end

    % Construct lagged controls
    n_lag  = min(h, n_lag_max);
    W_lag  = NaN(length(i_index), n_lag, 1+n_s);
    for i_lag = 1:n_lag
        W_lag(:, i_lag, 1) = time_shift(y, i_index, t_index, -i_lag);
        for i_s = 1:n_s
            W_lag(:, i_lag, 1+i_s) = time_shift(sX(:, i_s), i_index, t_index, -i_lag);
        end
    end
    W_lag  = reshape(W_lag, [length(i_index), (1+n_s)*n_lag]);

    % Prepare data
    d      = ~any(isnan([y_h, sX, W_lag, W]), 2);
    y_LP   = y_h(d, :);
    X_LP   = [sX(d, :), W_lag(d, :), W(d, :)];
    n_X    = size(X_LP, 2);
    dum_LP = FEs(d, :);

    % Prepare time-series indexes
    t_LP  = t_index(d);
    t_set = sort(unique(t_LP), 'ascend');
    T     = length(t_set);
    
    % Compute LP estimator    
    [b_LP, y_LP, X_LP]  = regress_HDFE(y_LP, X_LP, dum_LP);
    LP_estimate(h+1, :) = b_LP(1:n_s);

    % Compute score and hessian
    Xv_it = X_LP .* (y_LP - X_LP*b_LP);
    Xv_t  = zeros(T, n_X);
    for t = 1:T
        t_tmp      = (t_LP == t_set(t));
        Xv_t(t, :) = sum(Xv_it(t_tmp, :), 1);
    end
    XX    = (X_LP')*X_LP;

    % Compute t-LAHR standard error
    if (small_sample == true)

        % Compute Imbens-Kolesar small-sample refinement
        X_t    = zeros(T, n_X);
        for t = 1:T
            t_tmp     = (t_LP == t_set(t));
            X_t(t, :) = sum(X_LP(t_tmp, :), 1);
        end
        P0     = eye(T) - X_t*pinv((X_t')*X_t)*(X_t');
        Xv_var = ((Xv_t./sqrt(diag(P0)))')*(Xv_t./sqrt(diag(P0)));
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        G0     = zeros(T);
        XX0    = pinv((X_t')*X_t);
        for t = 1:T, G0(:, t) = P0(:, t)*X_t(t, :)*XX0(:, 1)/sqrt(P0(t, t)); end
        lam0   = eig(G0'*G0);

        % Store standard error and degrees of freedom
        LP_CI.SE(h+1, :) = sqrt(max(0, diag(b_var(1:n_s, 1:n_s))));
        LP_CI.df(h+1)    = (sum(lam0))^2/(sum(lam0.^2));

    else

        % Compute time-clustered sandwich formula
        Xv_var = (Xv_t')*Xv_t;
        b_var  = pinv(XX)*Xv_var*pinv(XX);

        % Store standard error and degrees of freedom
        LP_CI.SE(h+1, :) = sqrt(max(0, diag(b_var(1:n_s, 1:n_s))));
        LP_CI.df(h+1)    = Inf;

    end

    % Compute confidence intervals
    cv_tmp                = tinv(1-(1-0.90)/2, LP_CI.df(h+1));
    LP_CI.CI90(h+1, :, 1) = LP_estimate(h+1, :) - cv_tmp .* LP_CI.SE(h+1, :);     
    LP_CI.CI90(h+1, :, 2) = LP_estimate(h+1, :) + cv_tmp .* LP_CI.SE(h+1, :);     
    cv_tmp                = tinv(1-(1-0.95)/2, LP_CI.df(h+1));
    LP_CI.CI95(h+1, :, 1) = LP_estimate(h+1, :) - cv_tmp .* LP_CI.SE(h+1, :);     
    LP_CI.CI95(h+1, :, 2) = LP_estimate(h+1, :) + cv_tmp .* LP_CI.SE(h+1, :);     
    cv_tmp                = tinv(1-(1-0.99)/2, LP_CI.df(h+1));
    LP_CI.CI99(h+1, :, 1) = LP_estimate(h+1, :) - cv_tmp .* LP_CI.SE(h+1, :);     
    LP_CI.CI99(h+1, :, 2) = LP_estimate(h+1, :) + cv_tmp .* LP_CI.SE(h+1, :);     

end

end