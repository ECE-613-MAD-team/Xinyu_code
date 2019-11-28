function [mrc, src] = rational_check()

file = dir(fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'scores', '*.mat'));
metric_num = numel(file);
x = (1:6)';
org_img_num = 4744;
distortion_type = 4;
rcm = zeros(metric_num, org_img_num, distortion_type);
metric_name = cell(metric_num, 1);
for i = 1 : metric_num 
    metric_name{i} = file(i).name(1:end-4);
    q = load([fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'scores/'), file(i).name]);
    y_hat = q.y_hat;
    for j = 1 : org_img_num
        for k = 1 : distortion_type
            r = [ y_hat((j-1)*21+1); y_hat((j-1)*21+1 + (k-1)*5+1: (j-1)*21+1 + k*5 )];            
            rcm(i,j,k) = -corr(x, r, 'type', 'Kendall');
        end
    end
end

rcm = reshape(rcm, [metric_num, org_img_num*distortion_type]);
mrc = mean(rcm, 2);
src = std(rcm,[],2);

[smrc, idx] = sort(mrc,'ascend');
ssrc = src(idx);
v = [smrc ssrc];
figure; bar(v(:,1),'facecolor',[209, 75, 78]/255);
hold on; errorbar(v(:,1),v(:,2),'x','color',[18, 53, 85]/255);

xlim([0,17])
set(gca,'XTick',1:16)
set(gca,'XTickLabel',metric_name(idx));

rotate_ticklabel(gca,30); 
ylabel('Average KRCC');
% print('MySavedPlot','-dpdf');
