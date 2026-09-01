% plot_rde_bench_results.m
%
% Loads rde_bench_results.mat (produced by researchforge.interop.matlab_export
% / export_to_matlab.py) and reproduces, natively in MATLAB, the best-so-far
% accuracy comparison and Failure-Repetition-Rate / Negative-Transfer-Rate
% bar charts reported in the project documentation. Plain, portable
% MATLAB/Octave syntax -- no toolbox-specific functions.
%
% Usage (from this matlab/ folder):
%   plot_rde_bench_results

r = load('../matlab_data/rde_bench_results.mat');

task_names = {'digits', 'synthetic_ecg'};
conditions = {'full', 'no_memory', 'random'};
colors     = [0.17 0.37 0.54; 0.71 0.29 0.05; 0.53 0.53 0.53];
labels     = {'Full system', 'No memory', 'Random search'};

% ---- Figure 1: best-so-far accuracy curves ----
figure('Name', 'RDE-Bench: best-so-far accuracy', 'Color', 'w');
for ti = 1:numel(task_names)
    subplot(1, numel(task_names), ti);
    hold on;
    task = r.(task_names{ti});
    for ci = 1:numel(conditions)
        cond = task.(conditions{ci});
        curve = mean(cond.curves, 1);
        gens = (1:numel(curve)) - 2;   % index 1 = baseline trial (generation -1)
        plot(gens, curve, 'Color', colors(ci, :), 'LineWidth', 1.8, ...
            'DisplayName', labels{ci});
    end
    title(strrep(task_names{ti}, '_', ' '));
    xlabel('Generation'); ylabel('Best validation metric so far');
    grid on;
    if ti == 1
        legend('Location', 'southeast');
    end
end

% ---- Figure 2: FRR / NTR bar comparison ----
figure('Name', 'RDE-Bench: Failure Repetition Rate and Negative Transfer Rate', 'Color', 'w');
metrics       = {'failure_repetition_rate', 'negative_transfer_rate'};
metric_titles = {'Failure Repetition Rate (lower is better)', 'Negative Transfer Rate'};
for mi = 1:numel(metrics)
    subplot(1, numel(metrics), mi);
    vals = zeros(numel(task_names), numel(conditions));
    for ti = 1:numel(task_names)
        task = r.(task_names{ti});
        for ci = 1:numel(conditions)
            vals(ti, ci) = task.(conditions{ci}).(metrics{mi});
        end
    end
    b = bar(vals);
    for ci = 1:numel(conditions)
        set(b(ci), 'FaceColor', colors(ci, :));
    end
    set(gca, 'XTickLabel', strrep(task_names, '_', ' '));
    title(metric_titles{mi});
    ylabel('Rate');
    if mi == 1
        legend(labels, 'Location', 'northoutside', 'Orientation', 'horizontal');
    end
    grid on;
end

fprintf('Plotted results for tasks: %s\n', strjoin(task_names, ', '));
