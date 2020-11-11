% Plots the mean and uncertainty (variance) of time-series data
%
% Author: Julian Neri
% Affil: McGill University
% Date: May 1, 2020

function plot_uncertainty(time,mean,variance)
    mean = mean(:)'; variance = variance(:)';
    sigma2 = sqrt(variance);
    time_fill = [time, fliplr(time)];
    x_fill = [mean+sigma2, fliplr(mean-sigma2)];
    plot(time,mean,'g-','linewidth',2,'DisplayName','\mu');
    fill(time_fill, x_fill, 'r','linestyle','none','FaceAlpha', 0.2,'DisplayName','\mu \pm \sigma^2');
end