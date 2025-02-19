function [b, y_resid, X_resid] = regress_HDFE(y, X, FEs, tol, max_iter)
% REGRESS_HDFE: Efficient high-dimensional fixed effects regression.
%
% INPUTS:
%   y        - (N x 1) Dependent variable
%   X        - (N x K) Independent variables
%   FEs      - (N x G) Group identifiers (e.g., {firm, time, industry})
%   tol      - Convergence tolerance (default: 1e-8)
%   max_iter - Maximum number of iterations (default: 500)
%
% OUTPUTS:
%   b       - (K x 1) Estimated coefficients
%   Y_RESID - (N x 1) Dependent variable orthogonalized wrt fixed effects
%   X_RESID - (N x K) Independent variables orthogonalized wrt fixed effects
%
% Version: 2024 Jun 10

% Set default options
if nargin < 4, tol      = 1e-8; end
if nargin < 5, max_iter = 500;  end

% Recover number of regressors
n_X = size(X, 2);

% Convert identifiers to unique indices
n_FE      = size(FEs, 2);
group_ids = cell(1, n_FE);
for i_FE = 1:n_FE
    [~, ~, group_ids{i_FE}] = unique(FEs(:, i_FE));
end

% Initialize transformed variables
y_resid   = y;
X_resid   = X;
converged = false;
iter      = 0;

% Perform iterative residualization
while ~converged && (iter < max_iter)
    iter  = iter + 1;
    X_old = X_resid;
    y_old = y_resid;
    
    for i_FE = 1:n_FE
        % Compute means by fixed effect group
        group_mean_y = accumarray(group_ids{i_FE}, y_resid, [], @mean);
        group_mean_X = zeros(length(group_mean_y), n_X);
        for i_X = 1:n_X
            group_mean_X(:, i_X) = accumarray(group_ids{i_FE}, X_resid(:, i_X), [], @mean);
        end

        % Remove fixed effect from y and X
        y_resid = y_resid - group_mean_y(group_ids{i_FE});
        if (n_X > 0)
            X_resid = X_resid - group_mean_X(group_ids{i_FE}, :);
        end
    end
    
    % Check convergence
    if norm([y_resid, X_resid] - [y_old, X_old], 'fro') < tol
        converged = true;
    end
end

if ~converged
    warning('Fixed effects did not converge after %d iterations.', max_iter);
end

% Run OLS on residualized data
if (n_X > 0)
    b = X_resid\y_resid;
else
    b = [];
end

end