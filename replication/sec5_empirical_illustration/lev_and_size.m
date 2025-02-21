%%% EMPIRICAL ILLUSTRATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script performs an empirical analysis of firm investment responses
% to monetary policy shocks, quantifying the role and potential interaction
% between leverage and firm size by means of panel local projections.
%
% Results are reported in Section 5 of "Micro Responses to Macro Shocks" 
% by M. Almuzara and V. Sancibrian.
%
% The code takes as input a csv-file created by combining Compustat and CRSP 
% data with other sources following Ottonello and Winberry (2020, ECTA).
% It creates output directories for figures and matfiles.
%
% Version: 2024 June 10 - Matlab R2022a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Clear memory
clear
close all
clc

% Set filenames
main_name = 'lev_and_size';

% Set up directories
addpath('functions')
data_dir = ['indata' filesep];
fig_dir  = ['output' filesep 'figures' filesep];
mat_dir  = ['output' filesep 'matfiles' filesep];
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(mat_dir, 'dir'), mkdir(mat_dir); end

% Determine tasks
tasks                = struct();
tasks.synthetic_data = true;
tasks.SE_comparison  = true;

% Load data
data = readtable([data_dir 'panel_data.csv']);

% Set firm and time indexes
i_index   = data{:, 'i_index'};
t_year    = cellfun(@(c) str2double(c(1:4)), data{:, 't_index'});
t_quarter = cellfun(@(c) str2double(c(6)), data{:, 't_index'});
t_index   = 4*(t_year - min(t_year)) + t_quarter;
firm_dum  = categorical( data{:, 'i_index'} );
time_dum  = categorical( data{:, 't_index'} );
dummies   = [firm_dum, time_dum];

% Construct leverage and firm size
leverage  = (data{:, 'leverage'} - data{:, 'leverage_mean'});
leverage  = leverage/std(leverage, [], 1, 'omitnan');
firm_size = data{:, 'firm_size'};
firm_size = firm_size/std(firm_size, [], 1, 'omitnan');

% Construct regressand and regressors for panel local projection
y   = data{:, 'dlog_capital'};
X   = data{:, 'MP_shock'};
s   = [firm_size, leverage, firm_size.*leverage];
n_s = size(s, 2);

% Construct controls
W = time_shift([data{:, 'sales_growth'}, s .* data{:, 'GDP_growth'}, s .* data{:, 'unemployment_rate'}], i_index, t_index, -1);

% Set horizon and lead regressand
H   = 12;
y_h = [y, zeros(length(i_index), H)];
for h = 1:H
    y_h(:, h+1) = y_h(:, h) + time_shift(y, i_index, t_index, h);
end

% Set figure objects
white          = [  1,   1,   1];
black          = [  0,   0,   0];
gray           = [153, 153, 153]/255;
red            = [239,  65,  53]/255;
blue           = [  0,  85, 164]/255;
gold           = [218, 165,  32]/255;
green          = [ 50, 205,  50]/255;
font_square    = {'fontname', 'helvetica', 'fontsize', 40};
font_rectangle = {'fontname', 'helvetica', 'fontsize', 34};
fig_square     = {'units', 'inches', 'position', [0 0 14 14], ...
                  'paperunits', 'inches', 'papersize', [14 14], 'paperposition', [0 0 14 14]};
fig_rectangle  = {'units', 'inches', 'position', [0 0 24 10], ...
                  'paperunits', 'inches', 'papersize', [24 10], 'paperposition', [0 0 24 10]};


%% SYNTHETIC TIME SERIES

if (tasks.synthetic_data == true)

% Specify figure objects
c_list = [black; blue; red; gold; parula(max(0, n_s-3))];
m_list = [{'none'}; {'v'}; {'o'}; {'square'}; repmat({'none'}, [max(0, n_s-3), 1])];

% Initialize synthetic time series
t_TS = unique(t_index);
T    = length(t_TS);
Y_TS = NaN(T, n_s, H+1);
X_TS = NaN(T, 1);
w_TS = NaN(T, n_s, n_s, H+1);

% Compute synthetic regressor
for t = 1:T
    X_TS(t) = mean(X(t_index == t_TS(t)), 'omitnan');
end

% Run synthetic time series local projections
b_TS = zeros(H+1, n_s);
for h = 0:H
    tic_aux = cputime;

    % Prepare data
    d      = ~any(isnan([y_h(:, h+1), s, X, W]), 2);
    t_LP   = t_index(d);
    y_LP   = y_h(d, h+1);
    s_LP   = s(d, :);
    W_LP   = W(d, :);
    dum_LP = dummies(d, :);

    % Net out dummies from regressand
    [~, y_res, W_res] = regress_HDFE(y_LP, W_LP, dum_LP);
    y_res             = y_res - W_res*(W_res\y_res);
    s_res             = s_LP;

    % Compute synthetic regressand and weights
    for t = 1:T
        i_tmp              = (t_LP == t_TS(t));
        s_res(i_tmp, 1)    = s_res(i_tmp, 1) - mean(s_res(i_tmp, 1), 'omitnan');
        Y_TS(t, :, h+1)    = s_res(i_tmp, :)\y_res(i_tmp);
        w_TS(t, :, :, h+1) = (s_res(i_tmp, :)')*s_res(i_tmp, :);
    end

    % Compute estimates
    t_tmp        = ~isnan(X_TS);
    b_denom      = zeros(n_s, n_s);
    b_num        = zeros(1, n_s);
    for t = 1:T
        if (t_tmp(t) == true)
            b_denom = b_denom + X_TS(t)^2 * reshape(w_TS(t, :, :, h+1), [n_s, n_s]);
            b_num   = b_num + X_TS(t) * Y_TS(t, :, h+1) * reshape(w_TS(t, :, :, h+1), [n_s, n_s]);
        end
    end
    b_TS(h+1, :) = b_num/b_denom;

    fprintf('Synthetic time series LP - Horizon %2d/%d - %4.1f seconds\n', h, H, cputime-tic_aux)
end

% Run panel data local projections
b_PD = zeros(H+1, n_s);
for h = 0:H
    tic_aux = cputime;

    % Prepare data
    d      = ~any(isnan([y_h(:, h+1), s, X, W]), 2);
    y_LP   = y_h(d, h+1);
    X_LP   = [s(d, :) .* X(d), W(d, :)];
    dum_LP = dummies(d, :);

    % Compute estimates
    b_LP         = regress_HDFE(y_LP, X_LP, dum_LP);
    b_PD(h+1, :) = b_LP(1:n_s);

    fprintf('Panel data LP            - Horizon %2d/%d - %4.1f seconds\n', h, H, cputime-tic_aux)    
 end

% Store results
results      = struct();
results.t_TS = t_TS;
results.X_TS = X_TS;
results.Y_TS = Y_TS;
results.w_TS = w_TS;
results.b_TS = b_TS;
results.b_PD = b_PD;
save([mat_dir main_name '-synthetic.mat'], '-struct', 'results')

% Plot synthetic time series %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dates = datetime(min(t_year), 3*t_TS-1, 1);
d_tmp = ~any(isnan([X_TS, Y_TS(:, :, 1)]), 2);
X_tmp = (X_TS - mean(X_TS(d_tmp), 1))./std(X_TS(d_tmp));
Y_tmp = (Y_TS(:, :, 1) - mean(Y_TS(d_tmp, :, 1), 1))./std(Y_TS(d_tmp, :, 1), [], 1);
w_tmp = zeros(size(w_TS, 1), n_s);
for i_s = 1:n_s, w_tmp(:, i_s) = w_TS(:, i_s, i_s, 1); end
w_tmp = w_tmp./sum(w_tmp, 1, 'omitnan');

% Periods to shade
recession_dates = [datetime(1981,  7, 1), datetime(1982, 11, 1); ...
                   datetime(1990,  7, 1), datetime(1991,  3, 1); ...
                   datetime(2001,  3, 1), datetime(2001, 11, 1); ...
                   datetime(2007, 12, 1), datetime(2009,  6, 1); ...
                   datetime(2020,  2, 1), datetime(2020,  4, 1)];
ylim_tmp = [-4, 4];

%%% Data
fig0  = figure();
ax0   = axes();
fill(ax0, [recession_dates'; flipud(recession_dates')], repmat(kron(ylim_tmp, ones(1, 2))', 1, size(recession_dates, 1)), 0*white, 'facealpha', 0.1, 'linestyle', 'none')
hold('on')
plot0 = plot(ax0, dates, [X_tmp, Y_tmp]);
hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [min(dates(sum(w_tmp, 2) > 0)), max(dates(sum(w_tmp, 2) > 0))])
set(ax0, font_rectangle{:})
grid(ax0, 'on')
line(ax0, [min(dates(sum(w_tmp, 2) > 0)), max(dates(sum(w_tmp, 2) > 0))], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend_tmp = [{'X_t'}, cell(1, n_s)]; 
for i_s = 1:n_s, legend_tmp{i_s+1} = ['Y_{' num2str(i_s) 't}']; end
legend(plot0, legend_tmp, 'orientation', 'horizontal', 'location', 'north', 'box', 'off', font_rectangle{:})
set(plot0, {'linewidth'}, [{7}; repmat({4}, [n_s, 1])])
set(plot0, {'color'}, mat2cell(c_list(1:(n_s+1), :), ones(n_s+1, 1), 3))
set(plot0, {'linestyle'}, [{'-'}; {'-'}; {'-'}; {':'}; repmat({'-'}, [n_s-3, 1])])
set(plot0, {'marker'}, m_list(1:(n_s+1)))
set(plot0, {'markerfacecolor'}, mat2cell(c_list(1:(n_s+1), :), ones(n_s+1, 1), 3))
set(plot0, 'markersize', 9)

% Tune and save figure
set(fig0, fig_rectangle{:});
print(fig0, [fig_dir 'synthetic-data'], '-dpdf', '-vector')

%%% Weights
fig0  = figure();
ax0   = axes();
plot0 = plot(ax0, dates, w_tmp);

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
ylabel(ax0, 'Weights')
xlim(ax0, [min(dates(sum(w_tmp, 2) > 0)), max(dates(sum(w_tmp, 2) > 0))])
set(ax0, font_rectangle{:})
grid(ax0, 'on')

% Tune plot
legend(ax0, legend_tmp(2:(n_s+1)), 'orientation', 'horizontal', 'location', 'southeast', 'box', 'off', font_rectangle{:})
set(plot0, 'linewidth', 4)
set(plot0, {'color'}, mat2cell(c_list(2:(n_s+1), :), ones(n_s, 1), 3))
set(plot0, 'linestyle', '-')
set(plot0, {'marker'}, m_list(2:(n_s+1)))
set(plot0, {'markerfacecolor'}, mat2cell(c_list(2:(n_s+1), :), ones(n_s, 1), 3))
set(plot0, 'markersize', 9)

% Tune and save figure
set(fig0, fig_rectangle{:}); 
print(fig0, [fig_dir 'synthetic-weights'], '-dpdf', '-vector')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot point estimates %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i_s = 1:n_s
fig0  = figure();
ax0   = axes();
plot0 = plot(ax0, (0:H)', [b_TS(:, i_s), b_PD(:, i_s)]);
xlabel(ax0, 'h')
ylabel(ax0, ['LP estimates for s_{' num2str(i_s), '}']);

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [0, H])
set(ax0, font_square{:})
grid(ax0, 'on')
line(ax0, [0, H], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, {'Synthetic time series', 'Panel data'}, 'location', 'best', 'box', 'off', font_square{:})
set(plot0, {'linewidth'}, {4; 3})
set(plot0, {'color'}, {blue; red})
set(plot0, {'linestyle'}, {'-'; '-.'})
set(plot0, {'marker'}, {'x'; 'o'})
set(plot0, 'markersize', 15)

% Tune and save figure
set(fig0, fig_square{:}); 
print(fig0, [fig_dir 'synthetic-IRF_comparison-' num2str(i_s)], '-dpdf', '-vector')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end


%% STANDARD ERROR COMPARISON

if (tasks.SE_comparison == true)

% Specify confidence intervals
CI_names     = {'Unit-level', 'Two-way', 'DK98', 't-HR', 't-LAHR', 't-HAR'};
CI_short     = {'1W', '2W', 'DK98', 't-HR', 't-LAHR', 't-HAR'};
n_CI         = length(CI_names);
estimator_id = [1, 1, 1, 1, 2, 1];    
signif       = 0.9;

% Set figure objects
plot_CI    = [1, 2, 3, 5];
CI_lw      = repmat({7}, [n_CI, 1]);
CI_palette = flipud(turbo(n_CI)); CI_palette(end, :) = black;
CI_color   = mat2cell(CI_palette, ones(n_CI, 1), 3);
CI_style   = {':'; '--'; '-.'; ':'; '-'; '-.'};
CI_marker  = {'*'; 'o'; 'square'; 'x'; 'diamond'; '+'};

% Preallocate estimates and confidence intervals    
LP_data = zeros(H+1, 2, n_s);
SE_data = zeros(H+1, n_CI, n_s);
df_data = zeros(H+1, n_CI, n_s);

for h = 0:H
    tic_aux = cputime;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% no lag augmentation
    % Prepare data
    d      = ~any(isnan([y_h(:, h+1), s, X, W]), 2);
    y_LP   = y_h(d, h+1);
    X_LP   = [s(d, :).*X(d), W(d, :)];
    n_X    = size(X_LP, 2);
    dum_LP = dummies(d, :);

    % Prepare cross-section and time-series indexes
    i_LP  = i_index(d);
    t_LP  = t_index(d);
    i_set = sort(unique(i_LP), 'ascend');
    t_set = sort(unique(t_LP), 'ascend');
    N     = length(i_set);
    T     = length(t_set);

    % Compute LP estimator    
    [b_LP, y_LP, X_LP] = regress_HDFE(y_LP, X_LP, dum_LP);
    LP_data(h+1, 1, :) = b_LP(1:n_s);

    % Compute score and hessian
    Xv_it = X_LP .* (y_LP - X_LP*b_LP);
    Xv_i  = zeros(N, n_X);
    for i = 1:N
        i_tmp      = (i_LP == i_set(i));
        Xv_i(i, :) = sum(Xv_it(i_tmp, :), 1);
    end
    Xv_t  = zeros(T, n_X);
    for t = 1:T
        t_tmp      = (t_LP == t_set(t));
        Xv_t(t, :) = sum(Xv_it(t_tmp, :), 1);
    end
    XX    = (X_LP')*X_LP;

    % Compute unit-level standard error
    Xv_var = (Xv_i')*Xv_i;
    b_var  = pinv(XX)*Xv_var*pinv(XX);
    SE_data(h+1, 1, :) = sqrt(max(0, diag(b_var(1:n_s, 1:n_s))));
    df_data(h+1, 1, :) = Inf;

    % Compute two-way standard error
    Xv_var = (Xv_i')*Xv_i + (Xv_t')*Xv_t - (Xv_it')*Xv_it;
    b_var  = pinv(XX)*Xv_var*pinv(XX);
    SE_data(h+1, 2, :) = sqrt(max(0, diag(b_var(1:n_s, 1:n_s))));
    df_data(h+1, 2, :) = Inf;

    % Compute Driscoll-Kraay standard error
    Xv_var = (Xv_t')*Xv_t;
    n_NW   = min(h, ceil(0.75*T^(1/3)));
    for i_NW = 1:n_NW
        Xv_var = Xv_var + (Xv_t(1:(T-i_NW), :)')*Xv_t((i_NW+1):T, :) ...
            + (Xv_t((i_NW+1):T, :)')*Xv_t(1:(T-i_NW), :);
    end
    b_var  = pinv(XX)*Xv_var*pinv(XX);
    SE_data(h+1, 3, :) = sqrt(max(0, diag(b_var(1:n_s, 1:n_s))));
    df_data(h+1, 3, :) = Inf;
    
    % Compute t-HR standard error
    Xv_var = (Xv_t')*Xv_t;
    b_var  = pinv(XX)*Xv_var*pinv(XX);
    SE_data(h+1, 4, :) = sqrt(max(0, diag(b_var(1:n_s, 1:n_s))));
    df_data(h+1, 4, :) = Inf;

    % Compute t-HAR standard error
    n_EWC     = ceil(0.4*T^(2/3));
    cos_trans = sqrt(2)*cos(pi .* ((1:n_EWC)') .* (((1:T)-1/2)/T));
    Xv_var    = ((Xv_t')*(cos_trans')*cos_trans*Xv_t)/n_EWC;
    b_var     = pinv(XX)*Xv_var*pinv(XX);
    SE_data(h+1, 6, :) = sqrt(max(0, diag(b_var(1:n_s, 1:n_s))));
    df_data(h+1, 6, :) = n_EWC;

    %%% lag augmentation
    % Prepare data
    n_lag  = min(h, 2);
    W_lag  = NaN(length(i_index), n_lag, 1+n_s);
    for i_lag = 1:n_lag
        W_lag(:, i_lag, 1) = time_shift(y, i_index, t_index, -i_lag);
        for i_s = 1:n_s
            W_lag(:, i_lag, 1+i_s) = time_shift(s(:, i_s).*X, i_index, t_index, -i_lag);
        end
    end
    W_lag  = reshape(W_lag, [length(i_index), (1+n_s)*n_lag]);
    d      = ~any(isnan([y_h(:, h+1), s, X, W, W_lag]), 2);
    y_LP   = y_h(d, h+1);
    X_LP   = [s(d, :).*X(d), W_lag(d, :), W(d, :)];
    n_X    = size(X_LP, 2);
    dum_LP = dummies(d, :);

    % Prepare cross-section and time-series indexes
    i_LP  = i_index(d);
    t_LP  = t_index(d);
    i_set = sort(unique(i_LP), 'ascend');
    t_set = sort(unique(t_LP), 'ascend');
    N     = length(i_set);
    T     = length(t_set);

    % Compute LP estimator    
    [b_LP, y_LP, X_LP] = regress_HDFE(y_LP, X_LP, dum_LP);
    LP_data(h+1, 2, :) = b_LP(1:n_s);

    % Compute score and hessian
    Xv_it = X_LP .* (y_LP - X_LP*b_LP);
    Xv_t  = zeros(T, n_X);
    for t = 1:T
        t_tmp      = (t_LP == t_set(t));
        Xv_t(t, :) = sum(Xv_it(t_tmp, :), 1);
    end
    XX    = (X_LP')*X_LP;

    % Compute t-LAHR standard error
    X_t    = zeros(T, n_X);
    for t = 1:T
        t_tmp     = (t_LP == t_set(t));
        X_t(t, :) = sum(X_LP(t_tmp, :), 1);
    end
    P0     = eye(T) - X_t*pinv((X_t')*X_t)*(X_t');
    Xv_var = ((Xv_t./sqrt(diag(P0)))')*(Xv_t./sqrt(diag(P0)));
    b_var  = pinv(XX)*Xv_var*pinv(XX);
    SE_data(h+1, 5, :) = sqrt(max(0, diag(b_var(1:n_s, 1:n_s))));
    G0     = zeros(T);
    XX0    = pinv((X_t')*X_t);
    for t = 1:T, G0(:, t) = P0(:, t)*X_t(t, :)*XX0(:, 1)/sqrt(P0(t, t)); end
    lam0   = eig(G0'*G0);
    df_data(h+1, 5, :) = (sum(lam0))^2/(sum(lam0.^2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fprintf('Panel data LP and CI     - Horizon %2d/%d - %4.1f seconds\n', h, H, cputime-tic_aux)    
end

% Store results
results         = struct();
results.LP_data = LP_data;
results.SE_data = SE_data;
results.df_data = df_data;
save([mat_dir main_name '-comparison.mat'], '-struct', 'results')

% Plot point estimates %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i_s = 1:n_s
fig0  = figure();
ax0   = axes();
plot0 = plot(ax0, (0:H)', LP_data(:, :, i_s));
xlabel(ax0, 'h')
ylabel(ax0, 'LP estimates');

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [0, H])
set(ax0, font_square{:})
grid(ax0, 'on')
line(ax0, [0, H], [0, 0], 'linewidth', 1, 'color', black, 'linestyle', '--')

% Tune plot
legend(ax0, {'Standard', 'Lag-augmented'}, 'location', 'best', 'box', 'off', font_square{:})
set(plot0, {'linewidth'}, {4; 5})
set(plot0, {'color'}, {blue; red})
set(plot0, {'linestyle'}, {'-'; '-.'})

% Tune and save figure
set(fig0, fig_square{:}); 
print(fig0, [fig_dir 'comparison-estimates-' num2str(i_s)], '-dpdf', '-vector')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot standard errors %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i_s = 1:n_s
fig0     = figure();
ax0      = axes();
plot0    = plot(ax0, (0:H)', SE_data(:, :, i_s));
xlabel(ax0, 'h')
ylabel(ax0, 'Standard errors')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [0, H])
set(ax0, font_square{:})
grid(ax0, 'on')

% Tune plot
legend(ax0, CI_short, 'location', 'best', font_square{:}, 'box', 'off', 'numcolumns', 2, 'location', 'best')
set(plot0, {'linewidth'}, CI_lw)
set(plot0, {'color'}, CI_color)
set(plot0, {'linestyle'}, CI_style)
set(plot0, {'marker'}, CI_marker)
set(plot0, 'markersize', 15)

% Tune and save figure
set(fig0, fig_square{:});
print(fig0, [fig_dir 'comparison-standard_errors-' num2str(i_s)], '-dpdf', '-vector')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Plot confidence intervals %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i_s = 1:n_s
n_plot   = length(plot_CI);
LP_plot  = zeros(H+1, n_plot);
low_plot = zeros(H+1, n_plot);
upp_plot = zeros(H+1, n_plot);
for i_plot = 1:n_plot
    i_CI                = plot_CI(i_plot);
    LP_plot(:, i_plot)  = LP_data(:, estimator_id(i_CI), i_s);
    cv_tmp              = tinv(1-(1-signif)/2, df_data(:, i_CI, i_s));
    low_plot(:, i_plot) = LP_data(:, estimator_id(i_CI), i_s) - cv_tmp .* SE_data(:, i_CI, i_s);
    upp_plot(:, i_plot) = LP_data(:, estimator_id(i_CI), i_s) + cv_tmp .* SE_data(:, i_CI, i_s);
end
fig0  = figure();
ax0   = axes();
for i_plot = 1:n_plot
    i_CI = plot_CI(i_plot);
    fill(ax0, [0:H, fliplr(0:H)], [low_plot(:, i_plot)', fliplr(upp_plot(:, i_plot)')], ...
        CI_color{i_CI}, 'facealpha', 0.2, 'linestyle', CI_style(i_CI), 'linewidth', 5, 'edgecolor', CI_color{i_CI});
    hold('on')
end
plot0 = plot(ax0, (0:H)', LP_plot);
xlabel(ax0, 'h')
ylabel(ax0, ['LP estimates and ' num2str(100*signif) '% CIs']);

hold('off')

% Tune ax handle
set(ax0, 'LooseInset', get(ax0, 'TightInset'))
xlim(ax0, [0, H])
set(ax0, font_square{:})
grid(ax0, 'on')
line(ax0, [0, H], [0, 0], 'linewidth', 2, 'color', black, 'linestyle', '--')
legend(ax0, CI_short(plot_CI), 'location', 'southwest', 'box', 'off')

% Tune plot
set(plot0, {'linewidth'}, CI_lw(plot_CI))
set(plot0, {'color'}, CI_color(plot_CI))
set(plot0, {'linestyle'}, CI_style(plot_CI))

% Tune and save figure
set(fig0, fig_square{:});
print(fig0, [fig_dir 'comparison-CIs-' num2str(i_s)], '-dpdf', '-vector')
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end
