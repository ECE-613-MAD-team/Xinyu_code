function y = data_aggregation(algorithm_name)

file_path = fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'data', 'raw_scores/');
file = dir([file_path algorithm_name '/*.mat']);
y = zeros(length(file)*21,1);
for i = 1 : length(file)
    load([file_path algorithm_name '/' file(i).name]);
    y( (i-1)*21+1 : i*21 ) = y_segment;
end

