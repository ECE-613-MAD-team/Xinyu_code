function p = pristine_check()

file = dir(fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'scores', '*.mat'));
metric_num = numel(file);
p = zeros(metric_num, 1);
metric_name = cell(metric_num, 1);
for i = 1 : metric_num 
    metric_name{i} = file(i).name(1:end-4);
    q = load([fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'scores/'), file(i).name]);
    y_hat = q.y_hat(1:21:end);
    p(i) = std(y_hat) / mean(y_hat);
end

[sp, idx] = sort(p,'descend');
figure; bar(sp,'facecolor',[209, 75, 78]/255);
xlim([0,17])
set(gca,'XTick',1:16)
set(gca,'XTickLabel',metric_name(idx));

rotate_ticklabel(gca,30); 
ylabel('Coefficient of variation');
% print('MySavedPlot','-dpdf');
