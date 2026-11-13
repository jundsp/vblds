% PLOT_UNCERTAINTY  Plot a time series mean with a ±1 standard deviation band.
%
%   PLOT_UNCERTAINTY(time, mean, variance) draws the mean as a green line
%   and shades mean ± sqrt(variance) in red, on the current axes. Call
%   "hold on" first to add it to an existing plot.
%
% Inputs:
%   time      time axis                                       (1 x N)
%   mean      mean at each time                               (N values)
%   variance  variance at each time (not standard deviation)  (N values)
%
% Author: Julian Neri
% Affil: McGill University
% Date: May 1, 2020

function plot_uncertainty(time,mean,variance)
    mean = mean(:)'; variance = variance(:)';
    sigma = sqrt(variance);
    time_fill = [time, fliplr(time)];
    x_fill = [mean+sigma, fliplr(mean-sigma)];
    plot(time,mean,'g-','linewidth',2,'DisplayName','\mu');
    fill(time_fill, x_fill, 'r','linestyle','none','FaceAlpha', 0.2,'DisplayName','\mu \pm \sigma');
end