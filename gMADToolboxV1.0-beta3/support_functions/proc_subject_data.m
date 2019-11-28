function [ subject_info, status ] = proc_subject_data( subject_profile, total_num_pairs )
    % reg_status: 0 -> cancel -> exit the program
    %             1 -> new subject/old subject with his/her data removed -> exit to run training
    %             2 -> old subject with data available -> exit and continue test
    % subject_info: 1x9 cell
    %             1. subject index
    %             2. first name
    %             3. last name
    %             4. email
    %             5. gender
    %             6-8. birth year, month and date
    %             9. test pair completed
    
    % initialization
    status = 1;
    subject_info{1} = 0;
    subject_info(2:8) = subject_profile;
    subject_info{9} = 0;
    
    % first candidate
    if ~exist('./data/subject_list.mat', 'file')
        subject_info{1} = 1;
        subject_list = subject_info; %#ok
        save('./data/subject_list.mat', 'subject_list');
    else
        % second and subsequent candidates
        load('./data/subject_list.mat', 'subject_list');
        [subject_idx, reg_status, ctn_idx] = find_subject(subject_profile, subject_list, total_num_pairs); %#ok
        subject_info{1} = subject_idx;
        % the subject stopped in the middle of the test
        % start from where he/she stopped
        if (reg_status == 2)
            choice = questdlg('Your previous record is detected. Do you want to start from where you left?', ...
            'where to start?','Yes. Jump to where I left.','No. Start from the beginning.','Cancel',...
            'Yes. Jump to where I left.');
            % Handle response
            switch choice
                case 'Yes. Jump to where I left.'
                    status = 2;
                    subject_info{9} = ctn_idx;
                case 'No. Start from the beginning.'
                    subject_info{9} = 0;
                case 'Cancel'
                    status = 0;
            end
        % the candidate has completed the test, exit
        elseif (reg_status == 0)
            waitfor(msgbox('You completed the test!'));
            status = 0;
            return;
        % new candidate, start from the beginning
        else
            subject_list(subject_idx,:) = subject_info; %#ok
        end
    end
end

