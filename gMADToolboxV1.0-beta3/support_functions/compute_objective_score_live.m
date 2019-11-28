function sobj = compute_objective_score_live(algorithm_name, is_no_reference, is_color, LIVE_path)
    load([LIVE_path '/refnames_all.mat']);
    sobj = zeros(length(refnames_all),1); %#ok
    idx = 1;
    disp('Computing objective quality scores on the LIVE database for non-linear mapping (982 in total)...');
    % no reference IQA algorithm
    if is_no_reference
        if is_color
            for i = 1:227
                I = open_bitfield_bmp([LIVE_path '/jp2k/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:233
                I = open_bitfield_bmp([LIVE_path '/jpeg/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                I = open_bitfield_bmp([LIVE_path '/wn/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                I = open_bitfield_bmp([LIVE_path '/gblur/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                I = open_bitfield_bmp([LIVE_path '/fastfading/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
        else
            % grayscale only
            for i = 1:227
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/jp2k/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:233
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/jpeg/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/wn/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/gblur/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/fastfading/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
        end
    else
        % full reference IQA
        if is_color
            for i = 1:227
                refI = open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]);
                I = open_bitfield_bmp([LIVE_path '/jp2k/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:233
                refI = open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]);
                I = open_bitfield_bmp([LIVE_path '/jpeg/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                refI = open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]);
                I = open_bitfield_bmp([LIVE_path '/wn/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                refI = open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]);
                I = open_bitfield_bmp([LIVE_path '/gblur/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                refI = open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]);
                I = open_bitfield_bmp([LIVE_path '/fastfading/img' num2str(i) '.bmp']);
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
        else
            % grayscale only
            for i = 1:227
                refI = rgb2gray(open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]));
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/jp2k/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:233
                refI = rgb2gray(open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]));
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/jpeg/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                refI = rgb2gray(open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]));
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/wn/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                refI = rgb2gray(open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]));
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/gblur/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
            
            for i = 1:174
                refI = rgb2gray(open_bitfield_bmp([LIVE_path '/refimgs/' refnames_all{idx}]));
                I = rgb2gray(open_bitfield_bmp([LIVE_path '/fastfading/img' num2str(i) '.bmp']));
                sobj(idx) = feval(algorithm_name, refI, I);
                fprintf('Finished image %d / 982...\n', idx);
                idx = idx + 1;
            end
        end
    end
end

