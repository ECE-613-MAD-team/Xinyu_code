function [ ] = save_score(ref_name, comp_name, layer, subject_idx, subject_score, min_score, max_score, min_name, max_name)
    if exist(['./subjective_test_result/subject#' num2str(subject_idx) '/' ...
        ref_name '_' comp_name '_level#' num2str(layer) '.mat'], 'file')
        fs = load(['./subjective_test_result/subject#' num2str(subject_idx) '/' ...
            ref_name '_' comp_name '_level#' num2str(layer) '.mat']);
        final_score = fs.final_score;
        final_score(end+1,:) = [subject_score, min_score, max_score]; %#ok
        %by Qingbo
        final_name = fs.final_name;
        final_name(end+1,:) = {min_name,max_name}; %#ok
    else
        final_score = [subject_score, min_score, max_score]; %#ok
        %by Qingbo
        final_name = {min_name,max_name}; %#ok
    end
    save(['./subjective_test_result/subject#' num2str(subject_idx) '/' ...
        ref_name '_' comp_name '_level#' num2str(layer) '.mat'],...
        'final_score','final_name');
end

