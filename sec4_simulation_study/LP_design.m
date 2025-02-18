%%% MONTE CARLO SIMULATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version: 2024 February 08 - Matlab R2020a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear memory
clear
close all
clc

% Set filenamese
main_name = 'LP'; % LP model

% Set directories
fig_dir = ['output' filesep 'figures' filesep];
mat_dir = ['output' filesep 'matfiles' filesep];
tab_dir = ['output' filesep 'tables' filesep];

% Create directory for figures
if ~exist([fig_dir main_name], 'dir')
    mkdir([fig_dir main_name])
end
fig_dir = [fig_dir main_name filesep];

% Determine tasks
run_simulation = false;
create_figures = true;
set(0, 'DefaultFigureVisible', 'off')

% Set figure objects
white   = [  1,   1,   1];
black   = [  0,   0,   0];
red     = [239,  65,  53]/255;
blue    = [ 51,  51, 164]/255;
gold    = [255, 215,   0]/255;
figsize = {'units', 'inches', 'position', [0 0 14 14], ...
           'paperunits', 'inches', 'papersize', [14 14], 'paperposition', [0 0 14 14]};
font    = {'fontname', 'helvetica', 'fontsize', 40};

% Fix simulation settings
n_work     = 100;
n_MC       = 5000;
n_Y        = 1;    % dimension of micro outcome
n_S        = 1;    % dimension of individual characteristics
T_grid     = [30, 100];
N_grid     = 1000;
R2_grid    = [0.99, 0.66, 0.33];

for T = T_grid
for N = N_grid
for R2 = R2_grid    

% Close all active figures    
close all
    
% Set signal-noise
kappa = sqrt(N*(1-R2)/R2);

% Set maximum horizons
H_max = ceil(0.25*T);
L_bar = max(2*T, H_max);

% Specify data generating process
rho_s             = 0.5;
n_lambda          = 10;
Lambda_beta_AR    = [0.7; 0.3; 0.2; 0.1];
Lambda_beta_MA    = [0; 0];
Lambda_gamma_AR   = [0.7; 0.2; 0.1; -0.2];
Lambda_gamma_MA   = [0.2; -0.2];
Lambda_delta_AR   = [0.9; 0.3; 0.1; 0.1];
Lambda_delta_MA   = [0.5; 0.2];
par_beta          = struct();
par_beta.mean_AR  = Lambda_beta_AR;
par_beta.nobs_AR  = n_lambda;
par_beta.mean_MA  = Lambda_beta_MA;
par_beta.nobs_MA  = n_lambda;
par_gamma         = struct();
par_gamma.mean_AR = Lambda_gamma_AR; 
par_gamma.nobs_AR = n_lambda;
par_gamma.mean_MA = Lambda_gamma_MA;
par_gamma.nobs_MA = n_lambda;
par_delta         = struct();
par_delta.mean_AR = Lambda_delta_AR; 
par_delta.nobs_AR = n_lambda;
par_delta.mean_MA = Lambda_delta_MA;
par_delta.nobs_MA = n_lambda;


%% IRF DISTRIBUTIONS

% Set filename string
design_str = [main_name '_T' num2str(T) '_N' num2str(N) '_R' num2str(100*R2)];

% Create directory for figures
if ~exist([fig_dir design_str], 'dir')
    mkdir([fig_dir design_str])
end
fig_dir_local = [fig_dir design_str filesep];

% Reset random number generator
rng(2024)

% Simulate individual characteristics
n_dgp = 1e5;
C_mat = rho_s*ones(n_S+2); C_mat(logical(eye(n_S+2))) = 1;
s_dgp = ones(n_S+2, 1) + chol(C_mat, 'lower') * randn(n_S+2, n_dgp);

% Simulate heterogeneous parameters
beta_dgp  = zeros(n_Y, L_bar+1, n_dgp);
gamma_dgp = zeros(n_Y, L_bar+1, n_dgp);
delta_dgp = zeros(n_Y, n_Y, L_bar+1, n_dgp);
for i_Y = 1:n_Y
    for i = 1:n_dgp
        beta_tmp                  = draw_poly(par_beta, L_bar+1);
        beta_tmp                  = beta_tmp/sqrt(sum(beta_tmp.^2));
        beta_dgp(i_Y, :, i)       = s_dgp(1, i)*beta_tmp;
        gamma_tmp                 = draw_poly(par_gamma, L_bar+1);
        gamma_tmp                 = gamma_tmp/sqrt(sum(gamma_tmp.^2));
        gamma_dgp(i_Y, :, i)      = s_dgp(n_S+1, i)*gamma_tmp;
        delta_tmp                 = draw_poly(par_delta, L_bar+1);
        delta_tmp                 = delta_tmp/sqrt(sum(delta_tmp.^2));
        delta_dgp(i_Y, i_Y, :, i) = s_dgp(n_S+2, i)*delta_tmp;
    end
end

% Compute moments of distribution of heterogeneity
beta_mean  = reshape(mean(beta_dgp(1, :, :), 3), [L_bar+1, 1]);
gamma_mean = reshape(mean(gamma_dgp(1, :, :), 3), [L_bar+1, 1]);
delta_mean = reshape(mean(delta_dgp(1, 1, :, :), 4), [L_bar+1, 1]);
beta_proj  = reshape(beta_dgp(1, :, :), [L_bar+1, n_dgp])/(s_dgp(1:n_S, :)-mean(s_dgp(1:n_S, :), 2));

% Compute quantiles of IRFs
horizons    = (0:L_bar)';
quants      = [0.05, 0.15, 0.25, 0.5, 0.75, 0.85, 0.95];
n_quant     = length(quants);
quant_names = cell(1, (n_quant-1)/2); 
for j = 1:(n_quant-1)/2, quant_names{j} = [sprintf('%d', round(100*(quants(n_quant+1-j)-quants(j)))), '%']; end
beta_quant  = reshape(quantile(beta_dgp(1, :, :), quants, 3), [L_bar+1, n_quant]);
gamma_quant = reshape(quantile(gamma_dgp(1, :, :), quants, 3), [L_bar+1, n_quant]);
delta_quant = reshape(quantile(delta_dgp(1, 1, :, :), quants, 4), [L_bar+1, n_quant]);

if (create_figures == true)
% Plot quantiles of IRF distributions (beta) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [beta_quant(:, j)', fliplr(beta_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0 = plot(ax0, horizons, [beta_quant(:, (n_quant+1)/2), beta_mean]);
xlabel(ax0, 'h')
ylabel(ax0, '\beta_{ih}')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)]) 
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', ''}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {6; 5})
set(plot0, {'color'}, {blue; red})
set(plot0, {'linestyle'}, {'-'; '-.'})

% Tune and save figure
set(fig0, figsize{:}); 
print(fig0, [fig_dir_local 'true_IRF-beta'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot quantiles of IRF distributions (gamma) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [gamma_quant(:, j)', fliplr(gamma_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0 = plot(ax0, horizons, [gamma_quant(:, (n_quant+1)/2), gamma_mean]);
xlabel(ax0, 'h')
ylabel(ax0, '\gamma_{ih}')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', ''}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {6; 5})
set(plot0, {'color'}, {blue; red})
set(plot0, {'linestyle'}, {'-'; '-.'})

% Tune and save figure
set(fig0, figsize{:}); 
print(fig0, [fig_dir_local 'true_IRF-gamma'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot quantiles of IRF distributions (delta) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig0  = figure();
ax0   = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [delta_quant(:, j)', fliplr(delta_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0 = plot(ax0, horizons, [delta_quant(:, (n_quant+1)/2), delta_mean]);
xlabel(ax0, 'h')
ylabel(ax0, '\delta_{ih}')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', ''}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {6; 5})
set(plot0, {'color'}, {blue; red})
set(plot0, {'linestyle'}, {'-'; '-.'})

% Tune and save figure
set(fig0, figsize{:}); 
print(fig0, [fig_dir_local 'true_IRF-delta'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end


%% MONTE CARLO SIMULATION

% Specify confidence intervals
CI_names     = {'Unit-level', 'Two-way', 'DK98', 't-HR', 't-LAHR', 't-HAR'};
CI_short     = {'1W', '2W', 'DK98', 't-HR', 't-LAHR', 't-HAR'};
n_CI         = length(CI_names);
estimator_id = [1, 1, 1, 1, 2, 1];

if (run_simulation == true) || ~exist([mat_dir design_str '.mat'], 'file')

% Reset random number generator again
rng(2024)

% Preallocate estimands
estimand_mean    = NaN(H_max+1, n_MC);
estimand_proj    = NaN(H_max+1, n_MC);
estimand_proj_tv = NaN(H_max+1, n_MC);

% Preallocate estimators
LP_mean    = NaN(H_max+1, 2, n_MC);
LP_proj    = NaN(H_max+1, 2, n_MC);
LP_proj_tv = NaN(H_max+1, 2, n_MC);

% Preallocate confidence intervals
SE_mean    = NaN(H_max+1, n_CI, n_MC);
df_mean    = NaN(H_max+1, n_CI, n_MC);
SE_proj    = NaN(H_max+1, n_CI, n_MC);
df_proj    = NaN(H_max+1, n_CI, n_MC);
SE_proj_tv = NaN(H_max+1, n_CI, n_MC);
df_proj_tv = NaN(H_max+1, n_CI, n_MC);

par_pool = parpool(n_work);
parfor i_MC = 1:n_MC

    % Fix broadcast variables
    beta_mean_par = beta_mean;
    beta_proj_par = beta_proj;
    
    % Show progress
    message = sprintf('Data simulation          - Sample %d/%d\n', i_MC, n_MC);
    fprintf(message)
    
    % Simulate macro and micro shocks
    X = randn(L_bar+T, 1);
    Z = randn(L_bar+T, 1);
    U = randn(n_Y, L_bar+T, N);

    % Simulate individual characteristics
    C_mat = rho_s*ones(n_S+2); C_mat(logical(eye(n_S+2))) = 1;
    s     = ones(n_S+2, 1) + chol(C_mat, 'lower') * randn(n_S+2, N);
    s_tv  = NaN(n_S, T, N);
    for i = 1:N
        s_tv(:, :, i) = s(1:n_S, i) + randn(n_S, T);
    end

    % Simulate heterogeneous parameters
    mu    = randn(n_Y, N);
    beta  = zeros(n_Y, L_bar+1, N);
    gamma = zeros(n_Y, L_bar+1, N);
    delta = zeros(n_Y, n_Y, L_bar+1, N);
    for i_Y = 1:n_Y
        for i = 1:N
            beta_tmp              = draw_poly(par_beta, L_bar+1);
            beta_tmp              = beta_tmp/sqrt(sum(beta_tmp.^2));
            beta(i_Y, :, i)       = s(1, i)*beta_tmp;
            gamma_tmp             = draw_poly(par_gamma, L_bar+1);
            gamma_tmp             = gamma_tmp/sqrt(sum(gamma_tmp.^2));
            gamma(i_Y, :, i)      = s(n_S+1, i)*gamma_tmp;
            delta_tmp             = draw_poly(par_delta, L_bar+1);
            delta_tmp             = delta_tmp/sqrt(sum(delta_tmp.^2));
            delta(i_Y, i_Y, :, i) = s(n_S+2, i)*delta_tmp;
        end
    end

    % Compute data
    Y = NaN(n_Y, T, N);
    for i = 1:N
        for t = 1:T
            Y(:, t, i) = mu(:, i) + beta(:, :, i) * X(t+(L_bar:(-1):0), 1) + gamma(:, :, i) * Z(t+(L_bar:(-1):0), 1) ...
                       + kappa * reshape(delta(:, :, :, i), [n_Y, n_Y*(L_bar+1)]) * reshape(U(:, t+(L_bar:(-1):0), i), [n_Y*(L_bar+1), 1]);
        end
    end

    % Store simulated data
    X_sim    = X(L_bar+(1:T), 1);
    s_sim    = s(1:n_S, :);
    s_tv_sim = s_tv;
    Y_sim    = Y;

    % Compute and store estimands
    estimand_mean(:, i_MC)    = beta_mean_par(1:(H_max+1), 1);
    estimand_proj(:, i_MC)    = beta_proj_par(1:(H_max+1), 1);
    beta_proj_tv_tmp          = reshape(repmat(beta(1, 1:(H_max+1), :), [1, T, 1]), [H_max+1, T*N])/(reshape(s_tv, [n_S, T*N])-mean(reshape(s_tv, [n_S, T*N]), 2));
    estimand_proj_tv(:, i_MC) = beta_proj_tv_tmp(:, 1);
    
    % Set temporary storage for estimators
    LP_mean_par    = NaN(H_max+1, 2);
    LP_proj_par    = NaN(H_max+1, 2);
    LP_proj_tv_par = NaN(H_max+1, 2);
    
    % Set temporary storage for confidence intervals
    SE_mean_par    = NaN(H_max+1, n_CI);
    df_mean_par    = NaN(H_max+1, n_CI);
    SE_proj_par    = NaN(H_max+1, n_CI);
    df_proj_par    = NaN(H_max+1, n_CI);
    SE_proj_tv_par = NaN(H_max+1, n_CI);
    df_proj_tv_par = NaN(H_max+1, n_CI);
    
    for h = 0:H_max
        
        % Show progress
        message = sprintf('Estimation and inference - Sample %d/%d - Horizon %d/%d\n', i_MC, n_MC, h, H_max);
        fprintf(message)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %%% no lag augmentation
        % Compute LP estimator (1, unit FE)
        T_est = T-h;
        n_W   = 0;
        y     = reshape(Y_sim(1, (h+1):T, :), [T_est, N]);
        x     = repmat(X_sim(1:(T-h)), [1, N]);
        y     = y - mean(y, 1);
        x     = x - mean(x, 1);
        X     = x(:);
        b     = X\y(:);
        LP_mean_par(h+1, 1) = b(1);

        % Compute score and hessian
        n_X   = 1+n_W;
        Xv    = reshape(X .* (y(:) - X*b), [T_est, N, n_X]);
        Xv_i  = reshape(sum(Xv, 1), [N, n_X]);
        Xv_t  = reshape(sum(Xv, 2), [T_est, n_X]);
        Xv_it = reshape(Xv, [T_est*N, n_X]);
        XX    = (X')*X;
                
        % Compute unit-level standard error
        Xv_var = (Xv_i')*Xv_i;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_mean_par(h+1, 1) = sqrt(max(0, b_var(1)));
        df_mean_par(h+1, 1) = Inf;

        % Compute two-way standard error
        Xv_var = (Xv_i')*Xv_i + (Xv_t')*Xv_t - (Xv_it')*Xv_it;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_mean_par(h+1, 2) = sqrt(max(0, b_var(1)));
        df_mean_par(h+1, 2) = Inf;

        % Compute Driscoll-Kraay standard error
        Xv_var = (Xv_t')*Xv_t;
        n_NW   = min(h, ceil(0.75*T^(1/3)));
        for i_NW = 1:n_NW
            Xv_var = Xv_var + (Xv_t(1:(T_est-i_NW), :)')*Xv_t((i_NW+1):T_est, :) ...
                            + (Xv_t((i_NW+1):T_est, :)')*Xv_t(1:(T_est-i_NW), :);
        end
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_mean_par(h+1, 3) = sqrt(max(0, b_var(1)));
        df_mean_par(h+1, 3) = Inf;

        % Compute t-HR standard error
        Xv_var = (Xv_t')*Xv_t;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_mean_par(h+1, 4) = sqrt(max(0, b_var(1)));
        df_mean_par(h+1, 4) = Inf;

        % Compute t-HAR standard error
        n_EWC     = ceil(0.4*T_est^(2/3));
        cos_trans = sqrt(2)*cos(pi .* ((1:n_EWC)') .* (((1:T_est)-1/2)/T_est));
        Xv_var    = ((Xv_t')*(cos_trans')*cos_trans*Xv_t)/n_EWC;
        b_var     = pinv(XX)*Xv_var*pinv(XX);
        SE_mean_par(h+1, 6) = sqrt(max(0, b_var(1)));
        df_mean_par(h+1, 6) = n_EWC;
                
        %%% lag augmentation
        % Compute LP estimator (1, unit FE)
        n_lag = min(h, ceil((T-h)^(1/3)));
        T_est = T-h-n_lag;
        n_W   = (1+n_Y)*n_lag;
        y     = reshape(Y_sim(1, (h+n_lag+1):T, :), [T_est, N] );
        x     = repmat(X_sim((n_lag+1):(T-h)), [1, N]);
        W     = zeros(T_est, N, n_W);
        for i_lag = 1:n_lag
            for i_Y = 1:n_Y
                W(:, :, (n_Y+1)*(i_lag-1)+i_Y) = reshape(Y_sim(i_Y, ((n_lag+1):(T-h))-i_lag, :), [T_est, N]);
            end
            W(:, :, (n_Y+1)*i_lag) = repmat(X_sim(((n_lag+1):(T-h))-i_lag), [1, N]);
        end
        y     = y - mean(y, 1);
        x     = x - mean(x, 1);
        W     = W - mean(W, 1);
        X     = [x(:), reshape(W, [T_est*N, n_W])];
        b     = X\y(:);
        LP_mean_par(h+1, 2) = b(1);

        % Compute score and hessian
        n_X   = 1+n_W;
        Xv    = reshape(X .* (y(:) - X*b), [T_est, N, n_X]);
        Xv_t  = reshape(sum(Xv, 2), [T_est, n_X]);
        XX    = (X')*X;
                
        % Compute t-LAHR standard error
        if (T > 100)
            Xv_var = (Xv_t')*Xv_t;
            b_var  = pinv(XX)*Xv_var*pinv(XX);
            SE_mean_par(h+1, 5) = sqrt(max(0, b_var(1)));
            df_mean_par(h+1, 5) = Inf;
        else
            X_t    = reshape( mean(reshape(X, [T_est, N, n_X]), 2), [T_est, n_X]);
            P0     = eye(T_est) - X_t*pinv((X_t')*X_t)*(X_t');
            Xv_var = ((Xv_t./sqrt(diag(P0)))')*(Xv_t./sqrt(diag(P0)));
            b_var  = pinv(XX)*Xv_var*pinv(XX);
            SE_mean_par(h+1, 5) = sqrt(max(0, b_var(1)));
            G0     = zeros(T_est);
            XX0    = pinv((X_t')*X_t);
            for t = 1:T_est, G0(:, t) = P0(:, t)*X_t(t, :)*XX0(:, 1)/sqrt(P0(t, t)); end
            lam0   = eig(G0'*G0);
            df_mean_par(h+1, 5) = (sum(lam0))^2/(sum(lam0.^2));
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %%% no lag augmentation
        % Compute LP estimator (s_i, unit FE, time FE)
        T_est = T-h;
        n_W   = 0;
        y     = reshape(Y_sim(1, (h+1):T, :), [T_est, N] );
        x     = zeros(T_est, N, n_S);
        for i_S = 1:n_S
            x(:, :, i_S) = X_sim(1:(T-h)) .* s_sim(i_S, :);
        end
        y     = y - mean(y, 1) - mean(y, 2) + mean(y(:)); 
        x     = x - mean(x, 1) - mean(x, 2) + reshape(mean(reshape(x, T_est*N, n_S), 1), [1, 1, n_S]);
        X     = reshape(x, [T_est*N, n_S]);
        b     = X\y(:);
        LP_proj_par(h+1, 1) = b(1);

        % Compute score and hessian
        n_X   = n_S+n_W;
        Xv    = reshape(X .* (y(:) - X*b), [T_est, N, n_X]);
        Xv_i  = reshape(sum(Xv, 1), [N, n_X]);
        Xv_t  = reshape(sum(Xv, 2), [T_est, n_X]);
        Xv_it = reshape(Xv, [T_est*N, n_X]);
        XX    = (X')*X;
                
        % Compute unit-level standard error
        Xv_var = (Xv_i')*Xv_i;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_par(h+1, 1) = sqrt(max(0, b_var(1)));
        df_proj_par(h+1, 1) = Inf;

        % Compute two-way standard error
        Xv_var = (Xv_i')*Xv_i + (Xv_t')*Xv_t - (Xv_it')*Xv_it;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_par(h+1, 2) = sqrt(max(0, b_var(1)));
        df_proj_par(h+1, 2) = Inf;

        % Compute Driscoll-Kraay standard error
        Xv_var = (Xv_t')*Xv_t;
        n_NW   = min(h, ceil(0.75*T^(1/3)));
        for i_NW = 1:n_NW
            Xv_var = Xv_var + (Xv_t(1:(T_est-i_NW), :)')*Xv_t((i_NW+1):T_est, :) ...
                            + (Xv_t((i_NW+1):T_est, :)')*Xv_t(1:(T_est-i_NW), :);
        end
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_par(h+1, 3) = sqrt(max(0, b_var(1)));
        df_proj_par(h+1, 3) = Inf;

        % Compute t-HR standard error
        Xv_var = (Xv_t')*Xv_t;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_par(h+1, 4) = sqrt(max(0, b_var(1)));
        df_proj_par(h+1, 4) = Inf;        
        
        % Compute t-HAR standard error
        n_EWC     = ceil(0.4*T_est^(2/3));
        cos_trans = sqrt(2)*cos(pi .* ((1:n_EWC)') .* (((1:T_est)-1/2)/T_est));
        Xv_var    = ((Xv_t')*(cos_trans')*cos_trans*Xv_t)/n_EWC;
        b_var     = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_par(h+1, 6) = sqrt(max(0, b_var(1)));
        df_proj_par(h+1, 6) = n_EWC;

        %%% lag augmentation
        % Compute LP estimator (s_i, unit FE, time FE)
        n_lag = min(h, ceil((T-h)^(1/3)));
        T_est = T-h-n_lag;
        n_W   = (n_Y+n_S)*n_lag;
        y     = reshape(Y_sim(1, (h+n_lag+1):T, :), [T_est, N] );
        x     = zeros(T_est, N, n_S);
        for i_S = 1:n_S
            x(:, :, i_S) = X_sim((n_lag+1):(T-h)) .* s_sim(i_S, :);
        end
        W     = zeros(T_est, N, n_W);
        for i_lag = 1:n_lag
            for i_Y = 1:n_Y
                W(:, :, (n_Y+n_S)*(i_lag-1)+i_Y) = reshape(Y_sim(i_Y, ((n_lag+1):(T-h))-i_lag, :), [T_est, N]);
            end
            for i_S = 1:n_S
                W(:, :, (n_Y+n_S)*(i_lag-1)+n_Y+i_S) = X_sim(((n_lag+1):(T-h))-i_lag) .* s_sim(i_S, :);
            end
        end        
        y     = y - mean(y, 1) - mean(y, 2) + mean(y(:)); 
        x     = x - mean(x, 1) - mean(x, 2) + reshape(mean(reshape(x, T_est*N, n_S), 1), [1, 1, n_S]);
        W     = W - mean(W, 1) - mean(W, 2) + reshape(mean(reshape(W, T_est*N, n_W), 1), [1, 1, n_W]);
        X     = [reshape(x, [T_est*N, n_S]), reshape(W, [T_est*N, n_W])];
        b     = X\y(:);
        LP_proj_par(h+1, 2) = b(1);

        % Compute score and hessian
        n_X   = n_S+n_W;
        Xv    = reshape(X .* (y(:) - X*b), [T_est, N, n_X]);
        Xv_t  = reshape(sum(Xv, 2), [T_est, n_X]);
        XX    = (X')*X;

        % Compute t-LAHR standard error
        if (T > 100)
            Xv_var = (Xv_t')*Xv_t;
            b_var  = pinv(XX)*Xv_var*pinv(XX);
            SE_proj_par(h+1, 5) = sqrt(max(0, b_var(1)));
            df_proj_par(h+1, 5) = Inf;
        else
            X_t    = reshape( mean(reshape(X, [T_est, N, n_X]), 2), [T_est, n_X]);
            P0     = eye(T_est) - X_t*pinv((X_t')*X_t)*(X_t');
            Xv_var = ((Xv_t./sqrt(diag(P0)))')*(Xv_t./sqrt(diag(P0)));
            b_var  = pinv(XX)*Xv_var*pinv(XX);
            SE_proj_par(h+1, 5) = sqrt(max(0, b_var(1)));
            G0     = zeros(T_est);
            XX0    = pinv((X_t')*X_t);
            for t = 1:T_est, G0(:, t) = P0(:, t)*X_t(t, :)*XX0(:, 1)/sqrt(P0(t, t)); end
            lam0   = eig(G0'*G0);
            df_proj_par(h+1, 5) = (sum(lam0))^2/(sum(lam0.^2));
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        %%% no lag augmentation
        % Compute LP estimator (s_it, unit FE, time FE)
        T_est = T-h;
        n_W   = 0;
        y     = reshape(Y_sim(1, (h+1):T, :), [T_est, N] );
        x     = zeros(T_est, N, n_S);
        for i_S = 1:n_S
            x(:, :, i_S) = X_sim(1:(T-h)) .* reshape(s_tv_sim(i_S, 1:(T-h), :), [T_est, N]);
        end
        y     = y - mean(y, 1) - mean(y, 2) + mean(y(:)); 
        x     = x - mean(x, 1) - mean(x, 2) + reshape(mean(reshape(x, T_est*N, n_S), 1), [1, 1, n_S]);
        X     = reshape(x, [T_est*N, n_S]);
        b     = X\y(:);
        LP_proj_tv_par(h+1, 1) = b(1);

        % Compute score and hessian
        n_X   = n_S+n_W;
        Xv    = reshape(X .* (y(:) - X*b), [T_est, N, n_X]);
        Xv_i  = reshape(sum(Xv, 1), [N, n_X]);
        Xv_t  = reshape(sum(Xv, 2), [T_est, n_X]);
        Xv_it = reshape(Xv, [T_est*N, n_X]);
        XX    = (X')*X;
                
        % Compute unit-level standard error
        Xv_var = (Xv_i')*Xv_i;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_tv_par(h+1, 1) = sqrt(max(0, b_var(1)));
        df_proj_tv_par(h+1, 1) = Inf;

        % Compute two-way standard error
        Xv_var = (Xv_i')*Xv_i + (Xv_t')*Xv_t - (Xv_it')*Xv_it;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_tv_par(h+1, 2) = sqrt(max(0, b_var(1)));
        df_proj_tv_par(h+1, 2) = Inf;

        % Compute Driscoll-Kraay standard error
        Xv_var = (Xv_t')*Xv_t;
        n_NW   = min(h, ceil(0.75*T^(1/3)));
        for i_NW = 1:n_NW
            Xv_var = Xv_var + (Xv_t(1:(T_est-i_NW), :)')*Xv_t((i_NW+1):T_est, :) ...
                            + (Xv_t((i_NW+1):T_est, :)')*Xv_t(1:(T_est-i_NW), :);
        end
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_tv_par(h+1, 3) = sqrt(max(0, b_var(1)));
        df_proj_tv_par(h+1, 3) = Inf;

        % Compute t-HR standard error
        Xv_var = (Xv_t')*Xv_t;
        b_var  = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_tv_par(h+1, 4) = sqrt(max(0, b_var(1)));
        df_proj_tv_par(h+1, 4) = Inf;        
        
        % Compute t-HAR standard error
        n_EWC     = ceil(0.4*T_est^(2/3));
        cos_trans = sqrt(2)*cos(pi .* ((1:n_EWC)') .* (((1:T_est)-1/2)/T_est));
        Xv_var    = ((Xv_t')*(cos_trans')*cos_trans*Xv_t)/n_EWC;
        b_var     = pinv(XX)*Xv_var*pinv(XX);
        SE_proj_tv_par(h+1, 6) = sqrt(max(0, b_var(1)));
        df_proj_tv_par(h+1, 6) = n_EWC;
        
        %%% lag augmentation
        % Compute LP estimator (s_it, unit FE, time FE)
        n_lag = min(h, ceil((T-h)^(1/3)));
        T_est = T-h-n_lag;
        n_W   = (n_Y+n_S)*n_lag;
        y     = reshape(Y_sim(1, (h+n_lag+1):T, :), [T_est, N]);
        x     = zeros(T_est, N, n_S);
        for i_S = 1:n_S
            x(:, :, i_S) = X_sim((n_lag+1):(T-h)) .* reshape(s_tv_sim(i_S, (n_lag+1):(T-h), :), [T_est, N]);
        end
        W     = zeros(T_est, N, n_W);
        for i_lag = 1:n_lag
            for i_Y = 1:n_Y
                W(:, :, (n_Y+n_S)*(i_lag-1)+i_Y) = reshape(Y_sim(i_Y, ((n_lag+1):(T-h))-i_lag, :), [T_est, N]);
            end
            for i_S = 1:n_S
                W(:, :, (n_Y+n_S)*(i_lag-1)+n_Y+i_S) = X_sim(((n_lag+1):(T-h))-i_lag) .* reshape(s_tv_sim(i_S, ((n_lag+1):(T-h))-i_lag, :), [T_est, N]);
            end
        end        
        y     = y - mean(y, 1) - mean(y, 2) + mean(y(:)); 
        x     = x - mean(x, 1) - mean(x, 2) + reshape(mean(reshape(x, T_est*N, n_S), 1), [1, 1, n_S]);
        W     = W - mean(W, 1) - mean(W, 2) + reshape(mean(reshape(W, T_est*N, n_W), 1), [1, 1, n_W]);
        X     = [reshape(x, [T_est*N, n_S]), reshape(W, [T_est*N, n_W])];
        b     = X\y(:);
        LP_proj_tv_par(h+1, 2) = b(1);

        % Compute score and hessian
        n_X  = n_S+n_W;
        Xv   = reshape(X .* (y(:) - X*b), [T_est, N, n_X]);
        Xv_t = reshape(sum(Xv, 2), [T_est, n_X]);
        XX   = (X')*X;

        % Compute t-LAHR standard error
        if (T > 100)
            Xv_var = (Xv_t')*Xv_t;
            b_var  = pinv(XX)*Xv_var*pinv(XX);
            SE_proj_tv_par(h+1, 5) = sqrt(max(0, b_var(1)));
            df_proj_tv_par(h+1, 5) = Inf;
        else
            X_t    = reshape( mean(reshape(X, [T_est, N, n_X]), 2), [T_est, n_X]);
            P0     = eye(T_est) - X_t*pinv((X_t')*X_t)*(X_t');
            Xv_var = ((Xv_t./sqrt(diag(P0)))')*(Xv_t./sqrt(diag(P0)));
            b_var  = pinv(XX)*Xv_var*pinv(XX);
            SE_proj_tv_par(h+1, 5) = sqrt(max(0, b_var(1)));
            G0     = zeros(T_est);
            XX0    = pinv((X_t')*X_t);
            for t = 1:T_est, G0(:, t) = P0(:, t)*X_t(t, :)*XX0(:, 1)/sqrt(P0(t, t)); end
            lam0   = eig(G0'*G0);
            df_proj_tv_par(h+1, 5) = (sum(lam0))^2/(sum(lam0.^2));
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end

    % Save estimators
    LP_mean(:, :, i_MC)    = LP_mean_par;
    LP_proj(:, :, i_MC)    = LP_proj_par;
    LP_proj_tv(:, :, i_MC) = LP_proj_tv_par;
    
    % Save confidence intervals
    SE_mean(:, :, i_MC)    = SE_mean_par;
    df_mean(:, :, i_MC)    = df_mean_par;
    SE_proj(:, :, i_MC)    = SE_proj_par;
    df_proj(:, :, i_MC)    = df_proj_par;
    SE_proj_tv(:, :, i_MC) = SE_proj_tv_par;
    df_proj_tv(:, :, i_MC) = df_proj_tv_par;
    
end
delete(par_pool)

% Save estimands, estimators and confidence intervals
sim_obj                  = struct();
sim_obj.estimand_mean    = estimand_mean;
sim_obj.estimand_proj    = estimand_proj;
sim_obj.estimand_proj_tv = estimand_proj_tv;
sim_obj.LP_mean          = LP_mean;
sim_obj.LP_proj          = LP_proj;
sim_obj.LP_proj_tv       = LP_proj_tv;
sim_obj.CI_names         = CI_names;
sim_obj.estimator_id     = estimator_id;
sim_obj.SE_mean          = SE_mean;
sim_obj.df_mean          = df_mean;
sim_obj.SE_proj          = SE_proj;
sim_obj.df_proj          = df_proj;
sim_obj.SE_proj_tv       = SE_proj_tv;
sim_obj.df_proj_tv       = df_proj_tv;
save([mat_dir design_str '.mat'], '-struct', 'sim_obj')

end


%% COVERAGE AND EXPECTED LENGTH

% Recover simulation objects
sim_obj          = load([mat_dir design_str '.mat']);
estimand_mean    = sim_obj.estimand_mean;
estimand_proj    = sim_obj.estimand_proj;
estimand_proj_tv = sim_obj.estimand_proj_tv;
LP_mean          = sim_obj.LP_mean;
LP_proj          = sim_obj.LP_proj;
LP_proj_tv       = sim_obj.LP_proj_tv;
CI_names         = sim_obj.CI_names;
n_CI             = length(CI_names);
estimator_id     = sim_obj.estimator_id;
SE_mean          = sim_obj.SE_mean;
df_mean          = sim_obj.df_mean;
SE_proj          = sim_obj.SE_proj;
df_proj          = sim_obj.df_proj;
SE_proj_tv       = sim_obj.SE_proj_tv;
df_proj_tv       = sim_obj.df_proj_tv;

% Set significance level and critical values
signif = 0.90;

% Preallocate coverage and expected length
cov_mean    = NaN(n_CI, H_max+1);
len_mean    = NaN(n_CI, H_max+1);
cov_proj    = NaN(n_CI, H_max+1);
len_proj    = NaN(n_CI, H_max+1);
cov_proj_tv = NaN(n_CI, H_max+1);
len_proj_tv = NaN(n_CI, H_max+1);

% Compute summaries for confidence intervals for LP (1, unit FE)
for h = 0:H_max
    for i_CI = 1:n_CI
        LP_tmp   = reshape( LP_mean(h+1, estimator_id(i_CI), :), [1, n_MC]);
        SE_tmp   = reshape( SE_mean(h+1, i_CI, :), [1, n_MC]);
        cv_tmp   = tinv(1-(1-signif)/2, reshape( df_mean(h+1, i_CI, :), [1, n_MC]));
        low_tmp  = LP_tmp - SE_tmp .* cv_tmp;
        upp_tmp  = LP_tmp + SE_tmp .* cv_tmp;
        true_tmp = estimand_mean(h+1, :);
        cov_mean(i_CI, h+1) = mean( (low_tmp <= true_tmp) & (true_tmp <= upp_tmp) );
        len_mean(i_CI, h+1) = median(upp_tmp - low_tmp);
    end
end

% Compute summaries for confidence intervals for LP (s_i, unit FE, time FE)
for h = 0:H_max
    for i_CI = 1:n_CI
        LP_tmp   = reshape( LP_proj(h+1, estimator_id(i_CI), :), [1, n_MC]);
        SE_tmp   = reshape( SE_proj(h+1, i_CI, :), [1, n_MC]);
        cv_tmp   = tinv(1-(1-signif)/2, reshape( df_proj(h+1, i_CI, :), [1, n_MC]));
        low_tmp  = LP_tmp - SE_tmp .* cv_tmp;
        upp_tmp  = LP_tmp + SE_tmp .* cv_tmp;
        true_tmp = estimand_proj(h+1, :);
        cov_proj(i_CI, h+1) = mean( (low_tmp <= true_tmp) & (true_tmp <= upp_tmp) );
        len_proj(i_CI, h+1) = median(upp_tmp - low_tmp);
    end
end

% Compute summaries for confidence intervals for LP (s_it, unit FE, time FE)
for h = 0:H_max
    for i_CI = 1:n_CI
        LP_tmp   = reshape( LP_proj_tv(h+1, estimator_id(i_CI), :), [1, n_MC]);
        SE_tmp   = reshape( SE_proj_tv(h+1, i_CI, :), [1, n_MC]);
        cv_tmp   = tinv(1-(1-signif)/2, reshape( df_proj_tv(h+1, i_CI, :), [1, n_MC]));
        low_tmp  = LP_tmp - SE_tmp .* cv_tmp;
        upp_tmp  = LP_tmp + SE_tmp .* cv_tmp;        
        true_tmp = estimand_proj_tv(h+1, :);
        cov_proj_tv(i_CI, h+1) = mean( (low_tmp <= true_tmp) & (true_tmp <= upp_tmp) );
        len_proj_tv(i_CI, h+1) = median(upp_tmp - low_tmp);
    end
end

% Save coverage rates and expected length
sim_obj.cov_mean    = cov_mean;
sim_obj.len_mean    = len_mean;
sim_obj.cov_proj    = cov_proj;
sim_obj.len_proj    = len_proj;
sim_obj.cov_proj_tv = cov_proj_tv;
sim_obj.len_proj_tv = len_proj_tv;
save([mat_dir design_str '.mat'], '-struct', 'sim_obj')


%% TABLES AND FIGURES

% Set table objects
var_names = cell(1, 2*(H_max+1));
for h = 0:H_max
    var_names{1, h+1}           = sprintf('Cov%d', h);
    var_names{1, (H_max+1)+h+1} = sprintf('Len%d', h);
end

% Set figure objects
CI_lw         = repmat({7}, [n_CI, 1]);
CI_palette    = flipud(turbo(n_CI)); CI_palette(end, :) = black;
CI_color      = mat2cell(CI_palette, ones(n_CI, 1), 3);
CI_style      = {':'; '-'; '-.'; ':'; '-'; '-.'};
CI_marker     = {'*'; 'o'; 'square'; 'x'; 'diamond'; '+'};
CI_markersize = 12;
if (T >= 100)
    CI_markersize = 6;
end
if (T >= 250)
    CI_marker = repmat({'none'}, [n_CI, 1]);    
end

% Tabulate summaries for LP (1, unit FE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tab_summary = array2table([reshape(cov_mean, [n_CI, H_max+1]), len_mean], 'RowNames', CI_names, 'VariableNames', var_names);
disp(tab_summary)
writetable(tab_summary, [tab_dir design_str '.xlsx'], 'Sheet', 'mean', 'WriteRowNames', true, 'WriteVariableNames', true)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tabulate summaries for LP (s_i, unit FE, time FE) %%%%%%%%%%%%%%%%%%%%%%%
tab_summary = array2table([reshape(cov_proj, [n_CI, H_max+1]), len_proj], 'RowNames', CI_names, 'VariableNames', var_names);
disp(tab_summary)
writetable(tab_summary, [tab_dir design_str '.xlsx'], 'Sheet', 'proj', 'WriteRowNames', true, 'WriteVariableNames', true)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Tabulate summaries for LP (s_it, unit FE, time FE) %%%%%%%%%%%%%%%%%%%%%%
tab_summary = array2table([reshape(cov_proj_tv, [n_CI, H_max+1]), len_proj_tv], 'RowNames', CI_names, 'VariableNames', var_names);
disp(tab_summary)
writetable(tab_summary, [tab_dir design_str '.xlsx'], 'Sheet', 'proj_tv', 'WriteRowNames', true, 'WriteVariableNames', true)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (create_figures == true)
% Plot sampling distribution of LP (1, unit FE, no lags) %%%%%%%%%%%%%%%%%%
est_true  = mean(estimand_mean, 2);
est_mean  = mean(LP_mean(:, 1, :), 3);
est_quant = quantile(LP_mean(:, 1, :), quants, 3);
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [est_quant(:, j)', fliplr(est_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0    = plot(ax0, horizons, [est_quant(:, (n_quant+1)/2), est_mean, est_true]);
xlabel(ax0, 'h')
ylabel(ax0, 'Sampling distribution')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', 'Estimand'}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {5; 6; 6})
set(plot0, {'color'}, {blue; red; gold})
set(plot0, {'linestyle'}, {'-'; '-.'; ':'})

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'sampling_distribution-LP-mean'], '-dpdf', '-vector')
ylim_bias = get(ax0, 'ylim');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot sampling distribution of LP (1, unit FE, lags) %%%%%%%%%%%%%%%%%%%%%
est_true  = mean(estimand_mean, 2);
est_mean  = mean(LP_mean(:, 2, :), 3);
est_quant = quantile(LP_mean(:, 2, :), quants, 3);
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [est_quant(:, j)', fliplr(est_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0    = plot(ax0, horizons, [est_quant(:, (n_quant+1)/2), est_mean, est_true]);
xlabel(ax0, 'h')
ylabel(ax0, 'Sampling distribution')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
ylim(ax0, ylim_bias)
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', 'Estimand'}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {5; 6; 6})
set(plot0, {'color'}, {blue; red; gold})
set(plot0, {'linestyle'}, {'-'; '-.'; ':'})

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'sampling_distribution-LALP-mean'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot sampling distribution of LP (s_i, unit FE, time FE, no lags) %%%%%%%
est_true  = mean(estimand_proj, 2);
est_mean  = mean(LP_proj(:, 1, :), 3);
est_quant = quantile(LP_proj(:, 1, :), quants, 3);
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [est_quant(:, j)', fliplr(est_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0    = plot(ax0, horizons, [est_quant(:, (n_quant+1)/2), est_mean, est_true]);
xlabel(ax0, 'h')
ylabel(ax0, 'Sampling distribution')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', 'Estimand'}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {5; 6; 6})
set(plot0, {'color'}, {blue; red; gold})
set(plot0, {'linestyle'}, {'-'; '-.'; ':'})

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'sampling_distribution-LP-proj'], '-dpdf', '-vector')
ylim_bias = get(ax0, 'ylim');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot sampling distribution of LP (s_i, unit FE, time FE, lags) %%%%%%%%%%
est_true  = mean(estimand_proj, 2);
est_mean  = mean(LP_proj(:, 2, :), 3);
est_quant = quantile(LP_proj(:, 2, :), quants, 3);
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [est_quant(:, j)', fliplr(est_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0    = plot(ax0, horizons, [est_quant(:, (n_quant+1)/2), est_mean, est_true]);
xlabel(ax0, 'h')
ylabel(ax0, 'Sampling distribution')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
ylim(ax0, ylim_bias)
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', 'Estimand'}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {5; 6; 6})
set(plot0, {'color'}, {blue; red; gold})
set(plot0, {'linestyle'}, {'-'; '-.'; ':'})

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'sampling_distribution-LALP-proj'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot sampling distribution of LP (s_it, unit FE, time FE, no lags) %%%%%%
est_true  = mean(estimand_proj_tv, 2);
est_mean  = mean(LP_proj_tv(:, 1, :), 3);
est_quant = quantile(LP_proj_tv(:, 1, :), quants, 3);
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [est_quant(:, j)', fliplr(est_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0    = plot(ax0, horizons, [est_quant(:, (n_quant+1)/2), est_mean, est_true]);
xlabel(ax0, 'h')
ylabel(ax0, 'Sampling distribution')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', 'Estimand'}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {5; 6; 6})
set(plot0, {'color'}, {blue; red; gold})
set(plot0, {'linestyle'}, {'-'; '-.'; ':'})

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'sampling_distribution-LP-proj_tv'], '-dpdf', '-vector')
ylim_bias = get(ax0, 'ylim');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot sampling distribution of LP (s_it, unit FE, time FE, lags) %%%%%%%%%
est_true  = mean(estimand_proj_tv, 2);
est_mean  = mean(LP_proj_tv(:, 2, :), 3);
est_quant = quantile(LP_proj_tv(:, 2, :), quants, 3);
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
for j = 1:(n_quant-1)/2
    fill(ax0, [horizons', fliplr(horizons')], [est_quant(:, j)', fliplr(est_quant(:, n_quant+1-j)')], ...
        blue, 'facealpha', 0.1*j, 'linestyle', 'none', 'linewidth', 0.1, 'edgecolor', blue);
    hold('on')
end
plot0    = plot(ax0, horizons, [est_quant(:, (n_quant+1)/2), est_mean, est_true]);
xlabel(ax0, 'h')
ylabel(ax0, 'Sampling distribution')
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
ylim(ax0, ylim_bias)
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, [quant_names, {'Median', 'Mean', 'Estimand'}], 'location', 'best', 'box', 'off', font{:})
set(plot0, {'linewidth'}, {5; 6; 6})
set(plot0, {'color'}, {blue; red; gold})
set(plot0, {'linestyle'}, {'-'; '-.'; ':'})

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'sampling_distribution-LALP-proj_tv'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Plot coverage rates for LP (1, unit FE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
plot0    = plot(ax0, horizons, cov_mean');
xlabel(ax0, 'h')
ylabel(ax0, ['Coverage rate of ' num2str(100*signif) '% CI'])

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)]) 
yticks(ax0, 0:0.1:1)
ylim(ax0, [0, 1])
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [signif, signif], 'linewidth', 2, 'color', black, 'linestyle', ':')

% Tune plot
legend(ax0, CI_short, 'location', 'east', 'box', 'off', font{:}, 'numcolumns', 2)
set(plot0, {'linewidth'}, CI_lw)
set(plot0, {'color'}, CI_color)
set(plot0, {'linestyle'}, CI_style)
set(plot0, {'marker'}, CI_marker)
set(plot0, 'markersize', CI_markersize)

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'cov-mean'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot coverage rates for LP (s_i, unit FE, time FE) %%%%%%%%%%%%%%%%%%%%%%
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
plot0    = plot(ax0, horizons, cov_proj');
xlabel(ax0, 'h')
ylabel(ax0, ['Coverage rate of ' num2str(100*signif) '% CI'])

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)]) 
yticks(ax0, 0:0.1:1)
ylim(ax0, [0, 1])
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [signif, signif], 'linewidth', 2, 'color', black, 'linestyle', ':')

% Tune plot
legend(ax0, CI_short, 'location', 'east', 'box', 'off', font{:}, 'numcolumns', 2)
set(plot0, {'linewidth'}, CI_lw)
set(plot0, {'color'}, CI_color)
set(plot0, {'linestyle'}, CI_style)
set(plot0, {'marker'}, CI_marker)
set(plot0, 'markersize', CI_markersize)

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'cov-proj'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot coverage rates for LP (s_it, unit FE, time FE) %%%%%%%%%%%%%%%%%%%%%
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
plot0    = plot(ax0, horizons, cov_proj_tv');
xlabel(ax0, 'h')
ylabel(ax0, ['Coverage rate of ' num2str(100*signif) '% CI'])

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)]) 
yticks(ax0, 0:0.1:1)
ylim(ax0, [0, 1])
set(ax0, font{:})
grid(ax0, 'on')
line(ax0, [horizons(1), horizons(H_max)], [signif, signif], 'linewidth', 2, 'color', black, 'linestyle', ':')

% Tune plot
legend(ax0, CI_short, 'location', 'east', 'box', 'off', font{:}, 'numcolumns', 2)
set(plot0, {'linewidth'}, CI_lw)
set(plot0, {'color'}, CI_color)
set(plot0, {'linestyle'}, CI_style)
set(plot0, {'marker'}, CI_marker)
set(plot0, 'markersize', CI_markersize)

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'cov-proj_tv'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Plot length for LP (1, unit FE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
plot0    = plot(ax0, horizons, len_mean');
xlabel(ax0, 'h')
ylabel(ax0, ['Length of ' num2str(100*signif) '% CI'])

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
ylim(ax0, [0, 1.5*max(len_mean(:))])
set(ax0, font{:})
grid(ax0, 'on')

% Tune plot
legend(ax0, CI_short, 'location', 'northwest', 'box', 'off', font{:}, 'numcolumns', 2)
set(plot0, {'linewidth'}, CI_lw)
set(plot0, {'color'}, CI_color)
set(plot0, {'linestyle'}, CI_style)
set(plot0, {'marker'}, CI_marker)
set(plot0, 'markersize', CI_markersize)

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'len-mean'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot length for LP (s_i, unit FE, time FE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
plot0    = plot(ax0, horizons, len_proj');
xlabel(ax0, 'h')
ylabel(ax0, ['Length of ' num2str(100*signif) '% CI'])

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
ylim(ax0, [0, 1.5*max(len_proj(:))])
set(ax0, font{:})
grid(ax0, 'on')

% Tune plot
legend(ax0, CI_short, 'location', 'northwest', 'box', 'off', font{:}, 'numcolumns', 2)
set(plot0, {'linewidth'}, CI_lw)
set(plot0, {'color'}, CI_color)
set(plot0, {'linestyle'}, CI_style)
set(plot0, {'marker'}, CI_marker)
set(plot0, 'markersize', CI_markersize)

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'len-proj'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot length for LP (s_it, unit FE, time FE) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
horizons = (0:H_max)';
fig0     = figure();
ax0      = axes();
plot0    = plot(ax0, horizons, len_proj_tv');
xlabel(ax0, 'h')
ylabel(ax0, ['Length of ' num2str(100*signif) '% CI'])

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [horizons(1), horizons(H_max)])
ylim(ax0, [0, 1.5*max(len_proj_tv(:))])
set(ax0, font{:})
grid(ax0, 'on')

% Tune plot
legend(ax0, CI_short, 'location', 'northwest', 'box', 'off', font{:}, 'numcolumns', 2)
set(plot0, {'linewidth'}, CI_lw)
set(plot0, {'color'}, CI_color)
set(plot0, {'linestyle'}, CI_style)
set(plot0, {'marker'}, CI_marker)
set(plot0, 'markersize', CI_markersize)

% Tune and save figure
set(fig0, figsize{:});
print(fig0, [fig_dir_local 'len-proj_tv'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

end
end
end


%% LOCAL FUNCTIONS

function [coefs_trunc, coefs_AR, coefs_MA] = draw_poly(par, n_trunc)
% DRAW_POLY Generates AR and MA polynomial coefficients and their truncated moving average representation.
%
% SYNTAX:
%   [coefs_trunc, coefs_AR, coefs_MA] = draw_poly(par)
%   [coefs_trunc, coefs_AR, coefs_MA] = draw_poly(par, n_trunc)
%
% DESCRIPTION:
%   This function generates autoregressive (AR) and moving average (MA) polynomial coefficients
%   based on given mean roots and observational counts, and computes the truncated moving average
%   representation up to the specified number of lags.
%
% INPUTS:
%   par      - Structure containing the following optional fields:
%              * mean_AR (1xN double) : Mean AR roots (default: []).
%              * nobs_AR (scalar)     : Number of observations for AR coefficient draws (default: Inf).
%              * mean_MA (1xM double) : Mean MA roots (default: []).
%              * nobs_MA (scalar)     : Number of observations for MA coefficient draws (default: Inf).
%   n_trunc  - (Optional) Scalar defining the number of lags for the moving average representation
%              (default: 0).
%
% OUTPUTS:
%   coefs_trunc - (1xn_trunc double) Truncated moving average representation coefficients.
%   coefs_AR    - (1xN double) AR polynomial coefficients.
%   coefs_MA    - (1xM double) MA polynomial coefficients.
%
% NOTES:
%   - The function draws AR and MA coefficients from a Beta distribution if the number of observations
%     (nobs_AR or nobs_MA) is finite. Otherwise, the roots are directly assigned as mean_AR or mean_MA.
%   - The function uses a recursive procedure to compute the impulse response function (IRF), which
%     represents the truncated moving average coefficients.

% Extract parameters
if isfield(par, 'mean_AR'), mean_AR = par.mean_AR(:)'; else, mean_AR = [];  end
if isfield(par, 'nobs_AR'), nobs_AR = par.nobs_AR;     else, nobs_AR = Inf; end
if isfield(par, 'mean_MA'), mean_MA = par.mean_MA(:)'; else, mean_MA = [];  end
if isfield(par, 'nobs_MA'), nobs_MA = par.nobs_MA;     else, nobs_MA = Inf; end

% Draw AR coefficients
if isempty(mean_AR)
    coefs_AR = 0;
else
    if isinf(nobs_AR)
        roots_tmp = mean_AR;
    else
        roots_tmp = sign(mean_AR) .* betarnd(nobs_AR*abs(mean_AR), nobs_AR*(1-abs(mean_AR)));
    end
    poly_tmp = poly(roots_tmp);
    coefs_AR = -poly_tmp(2:end);
end

% Draw MA coefficients
if isempty(mean_MA)
    coefs_MA = 0;
else
    if isinf(nobs_MA)
        roots_tmp = mean_MA;
    else
        roots_tmp = sign(mean_MA).*betarnd(nobs_MA*abs(mean_MA), nobs_MA*(1-abs(mean_MA)));
    end
    poly_tmp = poly(roots_tmp);
    coefs_MA = poly_tmp(2:end);
end

% Compute moving average representation
if (nargin < 2), n_trunc = 0; end
coefs_trunc = NaN(1, n_trunc);
n_AR        = length(coefs_AR);
n_MA        = length(coefs_MA);
AR_aux      = [coefs_AR; eye(n_AR-1), zeros(n_AR-1, 1)];
MA_aux      = [1, coefs_MA];
IRF_aux     = zeros(n_AR, 1);
for i_trunc = 1:(n_MA+1)
    IRF_aux = AR_aux * IRF_aux + [MA_aux(i_trunc); zeros(n_AR-1, 1)];
    coefs_trunc(i_trunc) = IRF_aux(1);
end
for i_trunc = (n_MA+2):n_trunc
    IRF_aux = AR_aux * IRF_aux;
    coefs_trunc(i_trunc) = IRF_aux(1);
end

end