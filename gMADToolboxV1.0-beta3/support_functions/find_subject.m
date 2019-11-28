function [ sub_index, status, ctn_idx ] = find_subject( subject_info, subject_list, total_num_pairs )
    % status: 0 -> completed -> exit
    %         1 -> new subject
    %         2 -> old subject
    % match first name
    status = 1;
    ctn_idx = 0;
    rows = cellfun(@(x) strcmpi(x,subject_info{1}),subject_list(:,2));
    filtered_cell = subject_list(rows,:);
    % match last name
    rows = cellfun(@(x) strcmpi(x,subject_info{2}),filtered_cell(:,3));
    filtered_cell = filtered_cell(rows,:);
    % match gender
    rows = cellfun(@(x) strcmpi(x,subject_info{4}),filtered_cell(:,5));
    filtered_cell = filtered_cell(rows,:);
    % match birthday
    if (isempty(rows) == 0)
        [~, rows] = ismember(cell2mat(subject_info(5:7)),cell2mat(filtered_cell(:,6:8)),'rows');
        if (rows == 0)
            sub_index = max(cell2mat(subject_list(:,1))) + 1;
        else
            filtered_cell = filtered_cell(rows,:);
        end
    end
    
    if (isempty(rows))
        sub_index = max(cell2mat(subject_list(:,1))) + 1;
    else
        % the subject has completed the subjective test.
        % prevent a subject to conduct the test twice.
        if (total_num_pairs == filtered_cell{9})
            status = 0;
        % otherwise, the subject stopped during the test
        % continue from where he/she stopped
        else
            status = 2;
            sub_index = filtered_cell{1};
            ctn_idx = filtered_cell{9};
        end
    end
    
end

