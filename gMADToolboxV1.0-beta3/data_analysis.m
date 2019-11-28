function [] = data_analysis(test_result, algorithm_name)

addpath('./support_functions/');

data = test_result(2:end, 11:end);
data = cell2mat(data);

% outlier detection and subject removal
[pair_num, subject_num] = size(data);
pair_mean = mean(data,2);
pair_var = var(data, 0, 2);
m2 = pair_var .* (subject_num-1) ./ subject_num;
m4 = moment(data, 4, 2);
pair_kur = m4 ./ ((m2).^2 + eps);

cleaned_data = data;
T1 = repmat( ( pair_kur >= 2 ) .* ( pair_kur <= 4 ), [1, subject_num]);
ind1 = find( T1 == 1 );
dT11 = ( data - repmat( pair_mean, [1, subject_num] ) ) > 2 * repmat(pair_var.^0.5, [1, subject_num]);
dT12 = ( data - repmat( pair_mean, [1, subject_num] ) ) < -2 * repmat(pair_var.^0.5, [1, subject_num]);
ind11 = find(dT11 == 1);
ind12 = find(dT12 == 1);

T2 = 1 - T1;
ind2 = find( T2 == 1);
dT21 = ( data - repmat( pair_mean, [1, subject_num] ) ) > 20^0.5 * repmat(pair_var.^0.5, [1, subject_num]);
dT22 = ( data - repmat( pair_mean, [1, subject_num] ) ) <-20^0.5 * repmat(pair_var.^0.5, [1, subject_num]);
ind21 = find(dT21 == 1);
ind22 = find(dT22 == 1);

indP = union( intersect( ind1, ind11 ), intersect( ind1, ind12 ) );
indQ = union( intersect( ind2, ind21 ), intersect( ind2, ind22 ) );

X = zeros(pair_num, subject_num);
X(indP) = 1;
P = sum(X);
X = zeros(pair_num, subject_num);
X( indQ ) = 1;
Q = sum(X);
pind = ( P + Q > 0.05 * pair_num ) .* ( abs( (P - Q) ./ (P + Q) ) < 0.3 );
ind = find( repmat( pind, [pair_num, 1] ) == 1 );
cleaned_data( indP ) = nan;
cleaned_data( indQ ) = nan;
cleaned_data( ind ) = nan;

% aggressiveness and resistance matrices computation
preference_score = nanmean(cleaned_data, 2);
file = dir('./data/scores/*.mat');
metric_num = numel(file);
rank = 6;
preference_score = reshape(preference_score, [rank, metric_num, 2]);
preference_score = permute(preference_score, [2,1,3]);

A_n = preference_score(:, :, 1);
A_o = preference_score(:, :, 2);
R_n = 100 - abs(preference_score(:, :, 2));
R_o = 100 - abs(preference_score(:, :, 1));

% weight computation
weight_o = zeros(metric_num, rank);
for i = 1 : metric_num
    q = load(['./data/scores/' file(i).name ]);
    y_hat = q.y_hat;
    min_value = min(y_hat);
    max_value = max(y_hat);
    step = (max_value-min_value)/rank;
    check_points = ((min_value+(step*0.5)):step:(max_value-(step*0.5)))';
    range = [check_points-9, check_points+9];
    for j = 1 : rank 
        weight_o(i,j) = sum(y_hat>range(j,1) & y_hat<range(j,2));
    end
end
weight_o = (weight_o) ./ repmat( sum(weight_o, 2), [1, rank]);

weight_n = zeros(1, rank);
q = load(['./data/alg_score/' algorithm_name ]);
y_hat = q.y_hat;
min_value = min(y_hat);
max_value = max(y_hat);
step = (max_value-min_value)/rank;
check_points = ((min_value+(step*0.5)):step:(max_value-(step*0.5)))';
range = [check_points-9, check_points+9];
for j = 1 : rank 
    weight_n(j) = sum(y_hat>range(j,1) & y_hat<range(j,2));
end

weight_n = weight_n ./ sum(weight_n);

a_n = sum( A_n .* weight_o, 2 );
a_o = sum( A_o .* repmat(weight_n, [metric_num, 1]), 2 );
r_n = sum( R_n .* repmat(weight_n, [metric_num, 1]), 2 );
r_o = sum (R_o .* weight_o, 2);


M = load('./data/A');
A = zeros(metric_num+1);
A(1:end-1,1:end-1) = M.A;
A(1:end-1,end) = a_o;
A(end,1:end-1) = a_n';
M = load('./data/R');
R = zeros(metric_num+1);
R(1:end-1,1:end-1) = M.R;
R(1:end-1,end) = r_o;
R(end,1:end-1) = r_n';

% ranking by maximum likelihood
A = max(A, 0);
mla = scale_ml(A);
[~,a_rank]=sort(mla);
 
mlr = scale_ml(R);
[~,r_rank]=sort(mlr);

% ranking by eigenvector
% [V1, ~] = eig(A);
% eiga = abs(V1(:, 1));
% [~, a_rank]=sort(eiga);
% [V2, ~] = eig(R);
% eigr = abs(V2(:, 1));
% [~, r_rank] = sort(eigr);

% plot

metric_name = cell(metric_num,1);
for i = 1:metric_num
    metric_name{i}= file(i).name(1:end-4);
end
metric_name{end+1} = algorithm_name;
metric_num = metric_num + 1;


figure('Pos',[100 100 1300 400]);

bar(1:4:4*metric_num, mlr(a_rank),'BarWidth',0.2,'FaceColor','b');
hold on;
bar(2:4:4*metric_num, mla(a_rank),'BarWidth',0.2,'FaceColor','c');

xlim([0,4*metric_num])
set(gca, 'XTick', 1:4:4*metric_num)
set(gca, 'XTickLabel', metric_name(a_rank));
rotate_ticklabel(gca,30); 

ylabel('Global ranking score');
legend({'Resistance', 'Aggressiveness'},'Location','southeast');

rmpath('./support_functions/');
