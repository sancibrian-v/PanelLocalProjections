function y1 = time_shift(y0, i_index, t_index, L)
% TIME_SHIFT: Compute lag or lead of time series.
%
% Inputs:
%   Y0      - An n-by-m matrix of time series data, where n is the number of observations
%             and m is the number of variables.
%   i_index - An n-by-1 vector indicating the unit (individual) index for each observation.
%   t_index - An n-by-1 vector indicating the time index for each observation.
%   L       - An integer indicating the lead (if positive) or lag (if negative)
%             to be applied to the time series data.
%
% Outputs:
%   Y1 - An n-by-m matrix of time-shifted data, where each element is the lead or lag
%        of the corresponding element in y0, as specified by L.
%
% Version: 2024 Jun 10

% Loop over units
y1 = NaN(size(y0));
for i = unique(i_index)'

    % Extract unit data
    i_unit  = (i_index == i);
    t_unit  = t_index(i_unit);
    y0_unit = y0(i_unit, :);

    % Obtain data lag (negative L) or lead (positive L)
    [is_t, loc_t]    = ismember(t_unit+L, t_unit);
    y1_unit          = NaN(size(y0_unit));
    y1_unit(is_t, :) = y0_unit(loc_t(is_t), :);
    y1(i_unit, :)    = y1_unit;

end

end