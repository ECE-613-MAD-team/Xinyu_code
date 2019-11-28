function [] = run_subjective_test()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ==========================================================================%
% Group MAD Competition (gMAD), Version 1.0                                 %
% Copyright(c) 2016 Kede Ma, Qingbo Wu, Zhou Wang, Zhengfang Duanmu,        %
% Hongwei Yong, Lei Zhang and Hongliang Li                                  %
% All Rights Reserved.                                                      %
%                                                                           %
% --------------------------------------------------------------------------%
% Permission to use, copy, or modify this software and its documentation    %
% for educational and research purposes only and without fee is hereby      %
% granted, provided that this copyright notice and the original authors'    %
% names appear on all copies and supporting documentation. This program     %
% shall not be used, rewritten, or adapted as the basis of a commercial     %
% software or hardware product without first obtaining permission of the    %
% authors. The authors make no representations about the suitability of     %
% this software for any purpose. It is provided "as is" without express     %
% or implied warranty.                                                      %
%---------------------------------------------------------------------------%
% This is an implementation of subjective experiment for group MAD          %
% competition                                                               %
%                                                                           %
% Please refer to the following paper and the website for suggested         %
% usage                                                                     %
%                                                                           %
% Kede Ma, Qingbo Wu, Zhou Wang, Zhengfang Duanmu, Hongwei Yong, Lei Zhang  %
% and Hongliang Li, "Group MAD Competition: A New Methodology to Compare    %
% Objective Image Quality Models", CVPR, 2016.                              %
%                                                                           %
%                                                                           %
% Kindly report any suggestions or corrections to k29ma@uwaterloo.ca,       %
% zduanmu@uwaterloo.ca, or zhouwang@ieee.org                                %
%                                                                           %
%---------------------------------------------------------------------------%
% Subjective test execution                                                 %
% Usage:                                                                    %
%   1. Run the function.                                                    %
%       ex: >> run_subjective_test()                                        %
%   2. Once a rating window pops up, adjust the scroll bar to assign your   %
%   preference score to the image pair. Select a negative number if the     %
%   left image is preferred and vice versa. You can select the score        %
%   by clicking the desired point in the slider, dragging the scroll        %
%   bar or firstly moving the mouse into the slider region and then         %
%   rotate the mouse wheel.                                                 %
%   3. Once you have the preference score selected, click the 'next'        %
%   button.                                                                 %
%   4. After the subjective test is completed, the evaluation result is     %
%   stored in the 'subjective_test_result' folder. Each sub-directory       %
%   stores the result where the name of the directory as the reference      %
%   metric                                                                  %
%                                                                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% add path
    addpath('./support_functions/');
    
    %% load image names
    if ~exist('./data/test_config/image_indices.mat', 'file')
        error('Please run initialization first.');
    end
    ii = load('./data/test_config/image_indices.mat');
    total_num_pairs = length(ii.img_idx);
    
    if ~exist('./data/test_config/position_parity.mat', 'file')
        error('Please run initialization first.');
    end
    pp = load('./data/test_config/position_parity.mat');
    
    if ~exist('./result/test_result.mat', 'file')
        error('Please run initialization first.');
    end
    load('./result/test_result.mat');
    
    %% subject registration
    [reg_status, subject_profile] = registration_gui();
    if reg_status == -1
        rmpath('./support_functions/');
        return;
    else
        % subject_info: 1x9 cell
        % 1. subject index; 2. first name; 3. last name; 4. email; 5. gender; 6-8. birth year, month and date
        % 9. test pair completed
        [subject_info, status] = proc_subject_data(subject_profile, total_num_pairs);
    end
    
    %% run training session
    if (status == 1)
        run_training();
    elseif (status == 0)
        rmpath('./support_functions/');
        return;
    end
    
    col = find(strcmp(test_result(1,:), ['subject_' num2str(subject_info{1})])); %#ok
    if (isempty(col))
        col = size(test_result,2) + 1;
    end
    test_result{1,col} = ['subject_' num2str(subject_info{1})];
    
    %% start the test
    while subject_info{9} < total_num_pairs
        img_idx = ii.img_idx(subject_info{9}+1);
        % 1st half: current algorithm: competitor; other algorithm: anker
        % 2nd half: current algorithm: anker; other algorithm: competitor
        min_name = test_result{img_idx+1, 7};
        max_name = test_result{img_idx+1, 8};
        worst_img = open_bitfield_bmp(['./data/test_images/' min_name '.bmp']);
        best_img = open_bitfield_bmp(['./data/test_images/' max_name '.bmp']);

        if (pp.pos_parity(subject_info{9}+1) == 1)
            [subject_score, to_break] = image_compare_gui(worst_img, best_img, subject_info{9}, total_num_pairs);
        else
            [subject_score, to_break] = image_compare_gui(best_img, worst_img, subject_info{9}, total_num_pairs);
            subject_score = subject_score*(-1);
        end
        
        % the subject selected to take a break
        if (to_break == 1)
            break;
        end
        subject_info{9} = subject_info{9} + 1;
        test_result{img_idx+1,col} = subject_score; %#ok
    end
    
    load('./data/subject_list.mat');
    row = find(cell2mat(subject_list(:,1))==subject_info{1}); %#ok
    if (isempty(row))
        row = size(subject_list,1) + 1;
    end
    subject_list(row,:) = subject_info; %#ok
    save('./data/subject_list.mat', 'subject_list');
    save('./result/test_result.mat', 'test_result');
    
    clear all;
    f = figure;
    imshow('./data/thank_you.bmp');
    movegui(f,'center');

    rmpath('./support_functions/');
end