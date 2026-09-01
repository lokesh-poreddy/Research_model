% visualize_ecg_task.m
%
% Loads synthetic_ecg_task.mat (produced by researchforge.interop.matlab_export
% / export_to_matlab.py) and plots example normal vs. arrhythmic synthetic
% single-lead beats. Written in plain, portable MATLAB/Octave syntax
% (no toolbox-specific functions) so it runs in either environment.
%
% Usage (from this matlab/ folder):
%   visualize_ecg_task

data = load('../matlab_data/synthetic_ecg_task.mat');

X = data.X_train;
y = data.y_train(:);          % 0 = normal, 1 = arrhythmic
normal_idx = find(y == 0);
arrhythmic_idx = find(y == 1);
t = linspace(0, 1, size(X, 2));

figure('Name', 'ResearchForge-ECRM: synthetic ECG-style task', 'Color', 'w');

subplot(2, 1, 1);
hold on;
for k = 1:min(6, numel(normal_idx))
    plot(t, X(normal_idx(k), :), 'Color', [0.17 0.37 0.54], 'LineWidth', 1.1);
end
title(sprintf('Normal beats (label = 0) -- task: %s', data.task_name));
xlabel('Normalized time'); ylabel('Amplitude');
grid on;

subplot(2, 1, 2);
hold on;
for k = 1:min(6, numel(arrhythmic_idx))
    plot(t, X(arrhythmic_idx(k), :), 'Color', [0.71 0.29 0.05], 'LineWidth', 1.1);
end
title(sprintf('Arrhythmic beats (label = 1) -- target metric = %.2f', data.target_metric));
xlabel('Normalized time'); ylabel('Amplitude');
grid on;

fprintf('Loaded %d training examples (%d normal, %d arrhythmic), %d features each.\n', ...
    size(X, 1), numel(normal_idx), numel(arrhythmic_idx), size(X, 2));
